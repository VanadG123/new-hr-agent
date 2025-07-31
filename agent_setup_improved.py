import chromadb
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tempfile
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmployeeLookup:
    """Class to handle employee metadata operations"""

    def __init__(self, employee_metadata_file):
        self.df = None
        if employee_metadata_file is not None:
            try:
                # Handle StreamlitUploadedFile objects
                if hasattr(employee_metadata_file, 'getvalue'):
                    # It's a StreamlitUploadedFile, read from bytes
                    # Reset the file pointer to the beginning
                    employee_metadata_file.seek(0)
                    self.df = pd.read_csv(employee_metadata_file)
                    print(f"Successfully loaded CSV with {len(self.df)} employees")
                else:
                    # It's a file path, read directly
                    self.df = pd.read_csv(employee_metadata_file)
                    print(f"Successfully loaded CSV with {len(self.df)} employees")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                print(f"File type: {type(employee_metadata_file)}")
                print(f"File attributes: {dir(employee_metadata_file)}")
                self.df = None

    def find_employee(self, name: str) -> Optional[Dict[str, Any]]:
        """Find employee by name with fuzzy matching"""
        if self.df is None:
            return None

        # Exact match first
        exact_match = self.df[self.df['Employee Name'].str.lower() == name.lower()]
        if not exact_match.empty:
            return exact_match.iloc[0].to_dict()

        # Partial match - check if any part of the name matches
        name_parts = name.lower().split()
        for _, row in self.df.iterrows():
            employee_name_parts = row['Employee Name'].lower().split()
            if any(part in employee_name_parts for part in name_parts):
                return row.to_dict()

        return None

    def get_all_employees(self) -> list:
        """Get list of all employee names"""
        if self.df is None:
            print("DataFrame is None - no employees loaded")
            return []
        print(f"DataFrame shape: {self.df.shape}")
        print(f"DataFrame columns: {list(self.df.columns)}")
        if 'Employee Name' in self.df.columns:
            employees = self.df['Employee Name'].tolist()
            print(f"Found {len(employees)} employees: {employees[:3]}...")
            return employees
        else:
            print(f"Employee Name column not found. Available columns: {list(self.df.columns)}")
            return []

class SmartDocumentChunker:
    """Intelligent document chunking with context preservation"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_policy_documents(self, docs):
        """Chunk documents with enhanced metadata for better retrieval"""
        enhanced_chunks = []

        for doc in docs:
            # Identify document type from content
            content = doc.page_content.lower()
            doc_type = self._identify_doc_type(content)

            # Create chunks
            chunks = self.text_splitter.split_documents([doc])

            # Enhance chunks with metadata
            for chunk in chunks:
                # Add document type metadata
                chunk.metadata['doc_type'] = doc_type

                # Extract band information if present
                bands = re.findall(r'\bL[1-5]\b', chunk.page_content)
                if bands:
                    chunk.metadata['bands'] = ', '.join(list(set(bands)))

                # Extract team/department information
                teams = re.findall(r'\b(Engineering|Sales|HR|Finance|Operations|Ops/Support)\b', 
                                 chunk.page_content, re.IGNORECASE)
                if teams:
                    chunk.metadata['teams'] = ', '.join(list(set([t.lower() for t in teams])))

                # Extract policy section
                section = self._extract_section(chunk.page_content)
                if section:
                    chunk.metadata['section'] = section

                enhanced_chunks.append(chunk)

        return enhanced_chunks

    def _identify_doc_type(self, content: str) -> str:
        """Identify document type from content"""
        if 'leave' in content and 'work from' in content:
            return 'leave_policy'
        elif 'travel' in content and 'business travel' in content:
            return 'travel_policy'
        else:
            return 'unknown'

    def _extract_section(self, content: str) -> str:
        """Extract section name from content"""
        # Look for section headers with emojis or numbers
        section_patterns = [
            #r'[🎯💰🏖️🏢✈️🔒🚨✅📘🏷️🧠🧾🏡🧍❌✅🚨🔄📌]\s*\d*\.?\s*([^\n]+)',
            r'\d+\.\d+\s+([^\n]+)',
            r'\d+\.\s+([^\n]+)'
        ]

        for pattern in section_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1).strip()

        return 'general'

def setup_agent_executor(hr_leave_policy_file, hr_travel_policy_file, sample_offer_letter_file, employee_metadata_file):
    """
    Set up the agent executor with enhanced RAG capabilities for offer letter generation.

    Args:
        hr_leave_policy_file: Uploaded HR Leave & Work from Home Policy PDF
        hr_travel_policy_file: Uploaded HR Travel Policy PDF  
        sample_offer_letter_file: Uploaded Sample Offer Letter PDF (used as template, not embedded)
        employee_metadata_file: Uploaded Employee Metadata CSV

    Returns:
        agent_executor: Configured agent executor for generating offer letters
    """

    # 1. Initialize employee lookup
    employee_lookup = EmployeeLookup(employee_metadata_file)

    # 2. Load and process ONLY policy documents for embedding (exclude sample offer letter)
    docs = []
    policy_files = [
        (hr_leave_policy_file, "HR Leave & Work from Home Policy"),
        (hr_travel_policy_file, "HR Travel Policy")
    ]

    for policy_file, policy_name in policy_files:
        if policy_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(policy_file.getvalue())
                tmp_file_path = tmp_file.name

                try:
                    loader_pdf = PyPDFLoader(tmp_file_path)
                    loaded_docs = loader_pdf.load()
                    # Add source metadata
                    for doc in loaded_docs:
                        doc.metadata['source'] = policy_name
                    docs.extend(loaded_docs)
                finally:
                    os.unlink(tmp_file_path)

    # 3. Load sample offer letter as template (not for embedding)
    sample_template = ""
    if sample_offer_letter_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(sample_offer_letter_file.getvalue())
            tmp_file_path = tmp_file.name

            try:
                loader_pdf = PyPDFLoader(tmp_file_path)
                sample_docs = loader_pdf.load()
                sample_template = "\n".join([doc.page_content for doc in sample_docs])
            finally:
                os.unlink(tmp_file_path)

    # 4. Smart chunking with enhanced metadata
    chunker = SmartDocumentChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_policy_documents(docs)

    # 5. Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 6. Create vector store with ChromaDB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    )

    # 7. Set up LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    print(gemini_api_key)
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.3
    )

    # 8. Create comprehensive prompt template
    prompt_template = PromptTemplate(
        input_variables=["candidate_name", "employee_data", "leave_policy", "travel_policy", "sample_format"],
        template="""You are an expert HR professional tasked with generating a professional offer letter that strictly follows the company's format and policies.

EMPLOYEE INFORMATION:
{employee_data}

RELEVANT LEAVE POLICY:
{leave_policy}

RELEVANT TRAVEL POLICY:
{travel_policy}

SAMPLE FORMAT TO FOLLOW:
{sample_format}

INSTRUCTIONS:
1. Generate a complete offer letter for {candidate_name} using the EXACT format and structure shown in the sample
2. Use the employee's actual data for compensation, band, department, and location
3. Include band-specific leave entitlements from the leave policy
4. Include team-specific WFO requirements and WFH benefits (if applicable)
5. Include band-specific travel entitlements from the travel policy
6. Maintain the same professional tone and emoji usage as the sample
7. Use today's date and calculate appropriate joining date (typically 2-4 weeks from offer date)
8. Keep all legal sections (confidentiality, termination) identical to the sample format

IMPORTANT:
- Follow the sample structure EXACTLY
- Use real data from employee record, not placeholder values
- Ensure all band-specific benefits are accurate per policies
- Calculate performance bonus as percentage of base salary where applicable
- Include team-specific work arrangements

Generate the complete offer letter now:"""
    )

    # 9. Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)

    # 10. Create enhanced agent executor
    class EnhancedAgentExecutor:
        def __init__(self, vector_store, llm_chain, employee_lookup, sample_template):
            self.vector_store = vector_store
            self.llm_chain = llm_chain
            self.employee_lookup = employee_lookup
            self.sample_template = sample_template

        def invoke(self, input_dict):
            """Generate offer letter with intelligent policy retrieval"""
            try:
                query = input_dict.get("input", "")

                # Extract candidate name
                candidate_name = self._extract_candidate_name(query)
                if not candidate_name:
                    return {"output": "❌ Please specify a candidate name in your request."}

                # Find employee data
                employee_data = self.employee_lookup.find_employee(candidate_name)
                if not employee_data:
                    available_employees = self.employee_lookup.get_all_employees()
                    return {
                        "output": f"❌ Employee '{candidate_name}' not found.\n\n" +
                                f"Available employees: {', '.join(available_employees[:10])}..." +
                                f"\n\nTotal {len(available_employees)} employees in database."
                    }

                # Get relevant policies based on employee band and department
                leave_policy = self._get_relevant_policy("leave", employee_data)
                travel_policy = self._get_relevant_policy("travel", employee_data)

                # Format employee data
                formatted_employee_data = self._format_employee_data(employee_data)

                # Generate offer letter
                response = self.llm_chain.invoke({
                    "candidate_name": candidate_name,
                    "employee_data": formatted_employee_data,
                    "leave_policy": leave_policy,
                    "travel_policy": travel_policy,
                    "sample_format": self.sample_template
                })

                return {"output": response["text"]}

            except Exception as e:
                return {"output": f"❌ Error generating offer letter: {str(e)}"}

        def _extract_candidate_name(self, query: str) -> str:
            """Extract candidate name from query"""
            # Look for patterns like "for John Doe", "generate for Jane Smith", etc.
            patterns = [
                r"for\s+([A-Za-z\s]+?)(?:\.|$|\s+using|\s+with)",
                r"letter\s+for\s+([A-Za-z\s]+?)(?:\.|$|\s+using|\s+with)",
                r"generate.*?for\s+([A-Za-z\s]+?)(?:\.|$|\s+using|\s+with)"
            ]

            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    return match.group(1).strip()

            return ""

        def _get_relevant_policy(self, policy_type: str, employee_data: Dict) -> str:
            """Get relevant policy information based on employee data"""
            band = employee_data.get('Band', '')
            department = employee_data.get('Department', '').lower()

            # Create targeted search query
            search_queries = [
                f"{policy_type} {band}",
                f"{policy_type} {department}",
                f"{policy_type} policy {band} {department}"
            ]

            relevant_docs = []
            for query in search_queries:
                docs = self.vector_store.similarity_search(query, k=3)
                relevant_docs.extend(docs)

            # Remove duplicates and combine
            unique_docs = []
            seen_content = set()
            for doc in relevant_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)

            return "\n\n".join([doc.page_content for doc in unique_docs[:5]])

        def _format_employee_data(self, employee_data: Dict) -> str:
            """Format employee data for prompt"""
            return f"""
Name: {employee_data.get('Employee Name', 'N/A')}
Department: {employee_data.get('Department', 'N/A')}
Band: {employee_data.get('Band', 'N/A')}
Base Salary: ₹{employee_data.get('Base Salary (INR)', 0):,}
Performance Bonus: ₹{employee_data.get('Performance Bonus (INR)', 0):,}
Retention Bonus: ₹{employee_data.get('Retention Bonus (INR)', 0):,}
Total CTC: ₹{employee_data.get('Total CTC (INR)', 0):,}
Location: {employee_data.get('Location', 'N/A')}
Joining Date: {employee_data.get('Joining Date', 'N/A')}
"""

    # Return the enhanced agent executor
    return EnhancedAgentExecutor(vector_store, llm_chain, employee_lookup, sample_template)

# Default agent executor (will be updated when files are uploaded)
agent_executor = None
