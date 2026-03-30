#!/usr/bin/env python3
"""
Startup script for Medical Analysis Server
This script provides an easy way to start the server with proper configuration
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'requests', 'torch', 'numpy', 
        'sentence_transformers', 'langchain_community', 'faiss'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'server.py',
        'client.py', 
        'index.html',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All required files are present")
    return True

def check_rag_database():
    """Check if RAG database exists"""
    rag_path = Path('./rag')
    if not rag_path.exists():
        print("⚠️  RAG database directory './rag' not found")
        print("   The system may not work properly without medical knowledge base")
        return False
    
    print("✅ RAG database directory found")
    return True

def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server"""
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "server:app", 
        "--host", host, 
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"🚀 Starting server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def open_frontend():
    """Open the frontend in web browser"""
    frontend_path = Path('index.html').absolute()
    if frontend_path.exists():
        print(f"🌐 Opening frontend: file://{frontend_path}")
        webbrowser.open(f"file://{frontend_path}")
    else:
        print("❌ Frontend file (index.html) not found")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🏥 MEDICAL ANALYSIS SYSTEM - SERVER STARTUP")
    print("=" * 60)
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    
    if not check_files():
        print("\n❌ Setup incomplete. Please ensure all files are present.")
        sys.exit(1)
    
    if not check_dependencies():
        print("\n❌ Dependencies missing. Please install requirements.")
        sys.exit(1)
    
    check_rag_database()  # Warning only, not critical
    
    print("\n✅ System checks completed")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Medical Analysis Server Startup')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--no-reload', action='store_true', help='Disable auto-reload')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser')
    
    args = parser.parse_args()
    
    print(f"\n🔧 Configuration:")
    print(f"   Server: http://{args.host}:{args.port}")
    print(f"   Auto-reload: {'disabled' if args.no_reload else 'enabled'}")
    print(f"   Browser: {'disabled' if args.no_browser else 'will open'}")
    
    # Open frontend in browser (delayed)
    if not args.no_browser:
        import threading
        def delayed_open():
            time.sleep(2)  # Wait for server to start
            open_frontend()
        
        browser_thread = threading.Thread(target=delayed_open, daemon=True)
        browser_thread.start()
    
    # Start server
    print(f"\n🚀 Starting Medical Analysis Server...")
    print("   Endpoints:")
    print(f"   - Health Check: http://localhost:{args.port}/health")
    print(f"   - Analysis API: http://localhost:{args.port}/analyze")
    print(f"   - Documentation: http://localhost:{args.port}/docs")
    print()
    
    success = start_server(
        host=args.host, 
        port=args.port, 
        reload=not args.no_reload
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()