import streamlit as st
import pandas as pd
from agent_setup_improved import setup_agent_executor
import re
import sys

st.set_page_config(
    page_title="Agentic Offer Letter Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.write("Python executable used by Streamlit:")
st.write(sys.executable)



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📝 Agentic Offer Letter Generator</h1>', unsafe_allow_html=True)

# Sidebar for navigation and info
with st.sidebar:
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    **Steps to Generate Offer Letter:**
    1. Upload all required documents
    2. Verify employee data preview
    3. Enter candidate name
    4. Click generate

    **Features:**
    - ✅ Intelligent employee lookup
    - ✅ Band-specific policy retrieval
    - ✅ Exact format matching
    - ✅ Smart chunking & embedding
    """)

    st.markdown("### 📊 System Status")
    if 'agent_ready' in st.session_state and st.session_state.agent_ready:
        st.success("🟢 Agent Ready")
    else:
        st.warning("🟡 Upload files to activate agent")

# Initialize session state
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'employee_df' not in st.session_state:
    st.session_state.employee_df = None
if 'agent_ready' not in st.session_state:
    st.session_state.agent_ready = False

# File Upload Section
st.markdown('<h2 class="section-header">📁 Upload Required Documents</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Policy Documents**")
    hr_leave_policy = st.file_uploader(
        "1. HR Leave & Work from Home Policy (PDF)",
        type=['pdf'],
        help="Upload the company's leave and work from home policy document",
        key="leave_policy"
    )

    hr_travel_policy = st.file_uploader(
        "2. HR Travel Policy (PDF)", 
        type=['pdf'],
        help="Upload the company's travel policy document",
        key="travel_policy"
    )

with col2:
    st.markdown("**Templates & Data**")
    sample_offer_letter = st.file_uploader(
        "3. Sample Offer Letter (PDF)",
        type=['pdf'],
        help="Upload a sample offer letter template",
        key="sample_letter"
    )

    employee_metadata = st.file_uploader(
        "4. Employee Metadata (CSV)",
        type=['csv'],
        help="Upload CSV file containing employee data (salary, team, joining date, etc.)",
        key="employee_data"
    )

# File status display
st.markdown('<h3 class="section-header">📋 Upload Status</h3>', unsafe_allow_html=True)

uploaded_files = [hr_leave_policy, hr_travel_policy, sample_offer_letter, employee_metadata]
file_names = [
    "HR Leave & Work from Home Policy",
    "HR Travel Policy", 
    "Sample Offer Letter",
    "Employee Metadata"
]

status_cols = st.columns(4)
upload_complete = True

for i, (file, name) in enumerate(zip(uploaded_files, file_names)):
    with status_cols[i]:
        if file is not None:
            st.markdown(f'<div class="status-box success-box">✅ {name}<br><small>Uploaded</small></div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-box error-box">❌ {name}<br><small>Missing</small></div>', 
                       unsafe_allow_html=True)
            upload_complete = False

# Employee metadata preview and agent setup
if employee_metadata is not None:
    st.markdown('<h3 class="section-header">👥 Employee Database Preview</h3>', unsafe_allow_html=True)

    try:
        df = pd.read_csv(employee_metadata)
        st.session_state.employee_df = df

        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            st.metric("Departments", df['Department'].nunique())
        with col3:
            st.metric("Bands", df['Band'].nunique())
        with col4:
            avg_ctc = df['Total CTC (INR)'].mean()
            st.metric("Avg CTC", f"₹{avg_ctc:,.0f}")

        # Department and band distribution
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Department Distribution**")
            dept_counts = df['Department'].value_counts()
            st.bar_chart(dept_counts)

        with col2:
            st.markdown("**Band Distribution**")
            band_counts = df['Band'].value_counts().sort_index()
            st.bar_chart(band_counts)

        # Employee data table
        st.markdown("**Employee Records**")
        # Add search functionality
        search_term = st.text_input("🔍 Search employees by name or department:")

        if search_term:
            filtered_df = df[
                df['Employee Name'].str.contains(search_term, case=False, na=False) |
                df['Department'].str.contains(search_term, case=False, na=False)
            ]
            st.dataframe(filtered_df, use_container_width=True)
            st.caption(f"Showing {len(filtered_df)} of {len(df)} employees")
        else:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(df)} employees")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {e}")

# Setup agent executor when all files are uploaded
if upload_complete and not st.session_state.agent_ready:
    with st.spinner("🔧 Setting up AI agent with your documents..."):
        try:
            st.session_state.agent_executor = setup_agent_executor(
                hr_leave_policy, 
                hr_travel_policy, 
                sample_offer_letter, 
                employee_metadata
            )
            st.session_state.agent_ready = True
            st.success("✅ AI Agent is ready! You can now generate offer letters.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error setting up agent: {e}")
            st.info("💡 Make sure you have set your COGCACHE_API_KEY in the .env file.")

# Offer Letter Generation Section
st.markdown('<h2 class="section-header">🚀 Generate Offer Letter</h2>', unsafe_allow_html=True)

if st.session_state.agent_ready:
    col1, col2 = st.columns([2, 1])

    with col1:
        employee_name = st.text_input(
            "👤 Enter the full name of the candidate:",
            help="Type the exact name as it appears in the employee database, or a partial name for fuzzy matching"
        )

    with col2:
        st.markdown("**Quick Select**")
        if st.session_state.employee_df is not None:
            selected_employee = st.selectbox(
                "Or select from list:",
                options=[""] + st.session_state.employee_df['Employee Name'].tolist(),
                key="employee_select"
            )
            if selected_employee and not employee_name:
                employee_name = selected_employee

    # Generate button
    if st.button("🚀 Generate Offer Letter", type="primary", disabled=not bool(employee_name)):
        if employee_name:
            with st.spinner(f"🧠 Generating offer letter for {employee_name}..."):
                try:
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("🔍 Looking up employee data...")
                    progress_bar.progress(25)

                    status_text.text("📚 Retrieving relevant policies...")
                    progress_bar.progress(50)

                    status_text.text("✍️ Generating offer letter...")
                    progress_bar.progress(75)

                    # Generate the offer letter
                    response = st.session_state.agent_executor.invoke({
                        "input": f"Generate a complete offer letter for {employee_name} using the uploaded policies and metadata."
                    })

                    progress_bar.progress(100)
                    status_text.text("✅ Complete!")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Display the result
                    st.markdown("### 📄 Generated Offer Letter")

                    if response["output"].startswith("❌"):
                        st.error(response["output"])
                    else:
                        # Display the offer letter in a nice format
                        st.markdown(response["output"])

                        # Add download option
                        st.download_button(
                            label="📥 Download Offer Letter",
                            data=response["output"],
                            file_name=f"offer_letter_{employee_name.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                        # Success message
                        st.balloons()
                        st.success(f"✅ Offer letter generated successfully for {employee_name}!")

                except Exception as e:
                    st.error(f"❌ Error generating offer letter: {str(e)}")
                    st.info("💡 Please check your API key and try again.")
        else:
            st.warning("⚠️ Please enter a candidate's name.")
else:
    if not upload_complete:
        missing_files = [name for file, name in zip(uploaded_files, file_names) if file is None]
        st.warning(f"⚠️ Please upload all required files to activate the generator. Missing: {', '.join(missing_files)}")
    else:
        st.info("🔄 Agent is being set up... Please wait.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 <strong>Agentic Offer Letter Generator</strong> - Powered by AI RAG Technology</p>
    <p>Built with Streamlit, LangChain, ChromaDB, and OpenAI</p>
</div>
""", unsafe_allow_html=True)
