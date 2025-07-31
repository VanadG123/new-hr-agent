#!/bin/bash
# Setup script for Agentic Offer Letter Generator

echo "🚀 Setting up Agentic Offer Letter Generator..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env file from template
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file and add your GEMINI_API_KEY"
fi

# Create directories
echo "📁 Creating necessary directories..."
mkdir -p chroma_db

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your GEMINI_API_KEY"
echo "2. Run: streamlit run streamlit_app_improved.py"
echo ""
