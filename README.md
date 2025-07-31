# 📝 Agentic Offer Letter Generator

An intelligent AI-powered system that generates personalized offer letters for candidates based on HR policies, employee data, and company templates using advanced RAG (Retrieval-Augmented Generation) technology.

## 🌟 Features

### Core Capabilities
- **🤖 Intelligent Agent**: AI-powered offer letter generation with policy-aware content
- **📚 Smart RAG System**: Advanced document chunking and embedding with ChromaDB
- **🔍 Fuzzy Employee Lookup**: Find employees with partial name matching
- **📊 Band & Team-Specific Policies**: Automatic retrieval of relevant policies based on employee data
- **📋 Exact Format Matching**: Generates letters following company template precisely
- **🎯 Interactive UI**: Modern Streamlit interface with real-time feedback

### Technical Features
- **Intelligent Chunking**: Context-aware document splitting with metadata enhancement
- **Targeted Retrieval**: Band and department-specific policy extraction
- **Employee Database Integration**: CSV-based employee data with advanced search
- **Progress Tracking**: Real-time generation status with progress indicators
- **Error Handling**: Comprehensive validation and user-friendly error messages
- **Download Support**: Export generated letters in multiple formats

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Agent System   │    │  Vector Store   │
│                 │    │                 │    │   (ChromaDB)    │
│ • File Upload   │────│ • Employee      │────│ • Policy Docs   │
│ • Employee      │    │   Lookup        │    │ • Smart Chunks  │
│   Search        │    │ • Policy        │    │ • Embeddings    │
│ • Generation    │    │   Retrieval     │    │                 │
│   UI            │    │ • LLM Chain     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  OpenAI GPT-4   │
                       │   (CogCache)    │
                       │                 │
                       │ • Offer Letter  │
                       │   Generation    │
                       │ • Policy        │
                       │   Integration   │
                       └─────────────────┘
```

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+ 
- Gemini API Key 
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd agentic-offer-letter-generator
   ```

2. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure environment**
   ```bash
   # Edit .env file and add your API key
   nano .env
   # Add: GEMINI_API_KEY=your_actual_api_key_here
   ```

4. **Launch the application**
   ```bash
   streamlit run streamlit_app_improved.py
   ```

### Manual Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.template .env
   # Edit .env file with your API key
   ```

4. **Create directories**
   ```bash
   mkdir -p chroma_db
   ```

## 🚀 Usage Guide

### 1. Document Upload
Upload the following required files:
- **HR Leave & Work from Home Policy (PDF)**: Company leave policies
- **HR Travel Policy (PDF)**: Travel and expense policies  
- **Sample Offer Letter (PDF)**: Template for format reference
- **Employee Metadata (CSV)**: Employee database with salary and band info

### 2. Employee Data Format
Your CSV should contain these columns:
```csv
Employee Name,Department,Band,Base Salary (INR),Performance Bonus (INR),Retention Bonus (INR),Total CTC (INR),Location,Joining Date
```

### 3. Generate Offer Letters
1. Enter candidate name (supports fuzzy matching)
2. Or select from dropdown list
3. Click "Generate Offer Letter"
4. Download the generated letter

### 4. Features in Action
- **Smart Lookup**: "John" matches "John Smith" and "Johnny Doe"
- **Policy Integration**: Automatically includes band-specific leave entitlements
- **Team Policies**: Adds department-specific WFO requirements
- **Format Consistency**: Matches company template exactly

## 🔧 Configuration

### Environment Variables
```bash
# Required
GEMINI_API_KEY=your_api_key_here

# Optional (with defaults)
EMBEDDING_MODEL=all-MiniLM-L6-v2
TEMPERATURE=0.3
```

### Customization Options
- **Chunking Strategy**: Modify `SmartDocumentChunker` parameters
- **Prompt Template**: Customize the offer letter generation prompt
- **Employee Lookup**: Adjust fuzzy matching sensitivity
- **UI Theme**: Modify Streamlit CSS styling

## 🧠 AI Components

### RAG System
- **Document Loading**: PyPDFLoader for robust PDF parsing
- **Smart Chunking**: Context-aware splitting with metadata enrichment
- **Embeddings**: SentenceTransformers for semantic similarity
- **Vector Store**: ChromaDB for efficient similarity search
- **Retrieval**: Band and department-specific policy extraction

### LLM Integration
- **Model**: GPT-4 via CogCache for cost-effective access
- **Temperature**: Low (0.3) for consistent, professional output
- **Prompt Engineering**: Structured prompts with examples and constraints
- **Context Management**: Intelligent context window utilization

## 📊 System Components

### Core Classes

#### `EmployeeLookup`
- Fuzzy name matching algorithm
- Employee data validation
- Department and band filtering

#### `SmartDocumentChunker`
- Metadata-enhanced chunking
- Section identification
- Band/team extraction

#### `EnhancedAgentExecutor`
- Policy retrieval orchestration
- Employee data integration
- Offer letter generation pipeline

## 🔍 Advanced Features

### Intelligent Policy Retrieval
- **Band-Specific**: Retrieves policies relevant to employee's band level
- **Department-Specific**: Includes team-specific work arrangements
- **Context-Aware**: Maintains policy relationships and dependencies

### Document Processing Pipeline
1. **Upload & Validation**: File type and content validation
2. **Smart Chunking**: Context-preserving document segmentation
3. **Metadata Enhancement**: Band, team, and section tagging
4. **Embedding Generation**: Semantic vector creation
5. **Vector Storage**: ChromaDB persistence with metadata

### Generation Pipeline
1. **Employee Lookup**: Fuzzy matching with validation
2. **Policy Retrieval**: Targeted search based on employee profile
3. **Context Assembly**: Structured prompt with all relevant data
4. **LLM Generation**: GPT-4 powered content creation
5. **Post-Processing**: Format validation and enhancement

## 🛠️ Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Check .env file
cat .env
# Verify COGCACHE_API_KEY is set correctly
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### ChromaDB Issues
```bash
# Clear vector database
rm -rf chroma_db/
mkdir chroma_db
```

#### PDF Loading Issues
- Ensure PDFs are not password-protected
- Check file size limits (< 10MB recommended)
- Verify PDF format compatibility

### Performance Optimization
- **Chunking**: Adjust chunk size based on document complexity
- **Embeddings**: Use GPU if available for faster processing
- **Caching**: ChromaDB persists embeddings between sessions
- **Batch Processing**: Generate multiple letters efficiently

## 📈 Future Enhancements

### Planned Features
- **Multi-Language Support**: Generate offers in multiple languages
- **Template Variants**: Support for different offer letter templates
- **Approval Workflow**: Integration with approval systems
- **Email Integration**: Direct sending to candidates
- **Analytics Dashboard**: Generation metrics and insights

### Technical Improvements
- **Advanced RAG**: Implement hybrid search with keywords + semantic
- **Model Fine-Tuning**: Company-specific model adaptation
- **Real-Time Updates**: Dynamic policy updates without restart
- **API Endpoints**: RESTful API for system integration

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check troubleshooting section
- Review logs for error details

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **ChromaDB**: For vector storage capabilities
- **Streamlit**: For the interactive UI framework
- **Google**: For the underlying language model

---

Built with ❤️ for modern HR teams seeking efficiency and consistency in offer letter generation.
