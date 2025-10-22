"""
FAIR-LLM Installation Verification Script
Checks that all requirements are properly installed
"""
import sys
import os
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for prettier output (pip install rich)")

def print_header():
    """Print a nice header."""
    if RICH_AVAILABLE:
        console.print(Panel.fit("🚀 FAIR-LLM Installation Verification", style="bold blue"))
    else:
        print("=" * 60)
        print(" FAIR-LLM Installation Verification ".center(60))
        print("=" * 60)

def check_requirements_file():
    """Check if requirements.txt exists and parse it."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return None
    
    requirements = []
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Parse package name from requirement line
                pkg_name = line.split(">")[0].split("=")[0].split("[")[0].strip()
                requirements.append(pkg_name)
    return requirements

def check_python_version():
    """Check Python version."""
    python_version = sys.version_info
    py_version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]📐 Python Environment:[/bold cyan]")
        console.print(f"   Python version: {py_version_str}")
        console.print(f"   Python path: {sys.executable}")
    else:
        print(f"\n📐 Python Environment:")
        print(f"   Python version: {py_version_str}")
        print(f"   Python path: {sys.executable}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        return False, "Python 3.8+ is required"
    return True, None

def check_package_installation(requirements):
    """Check installation of all required packages."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]📦 Package Installation Status:[/bold cyan]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Package", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Version", style="green")
        table.add_column("Notes", style="yellow")
    else:
        print(f"\n📦 Package Installation Status:")
    
    all_good = True
    results = []
    
    # Define package info
    package_info = {
        'fair-llm': ('Core framework', True),
        'python-dotenv': ('Environment management', False),
        'rich': ('Terminal formatting', False),
        'anthropic': ('Claude AI integration', False),
        'faiss-cpu': ('Vector search', False),
        'seaborn': ('Data visualization', False),
        'pytest': ('Testing framework', False),
    }
    
    for pkg_name, (description, required) in package_info.items():
        try:
            pkg_version = version(pkg_name)
            
            # Special handling for fair-llm - just check if it's installed
            if pkg_name == 'fair-llm':
                # Package is installed, that's what matters
                status = "✅"
                notes = description
            else:
                # For other packages, try to import them
                import_name = pkg_name.replace('-', '_')
                if pkg_name == 'python-dotenv':
                    import_name = 'dotenv'
                elif pkg_name == 'faiss-cpu':
                    import_name = 'faiss'
                
                try:
                    __import__(import_name)
                    status = "✅"
                    notes = description
                except ImportError:
                    status = "⚠️"
                    notes = f"{description} (installed but can't import)"
            
            results.append((pkg_name, status, pkg_version, notes))
            
        except PackageNotFoundError:
            if required:
                status = "❌"
                notes = f"{description} (REQUIRED)"
                all_good = False
            else:
                status = "○"
                notes = f"{description} (optional)"
            results.append((pkg_name, status, "Not installed", notes))
    
    # Display results
    if RICH_AVAILABLE:
        for pkg_name, status, pkg_version, notes in results:
            table.add_row(pkg_name, status, pkg_version, notes)
        console.print(table)
    else:
        for pkg_name, status, pkg_version, notes in results:
            print(f"   {status} {pkg_name:<20} {pkg_version:<15} {notes}")
    
    return all_good

def check_fairlib_components():
    """Check if fairlib components are importable."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]🔧 FAIR-LLM Components:[/bold cyan]")
    else:
        print(f"\n🔧 FAIR-LLM Components:")
    
    # The package might be installed as fair-llm but imported as fair_llm or fairlib
    # Try to determine the correct import name
    possible_modules = ['fairlib', 'fair_llm']
    base_module = None
    
    for module_name in possible_modules:
        try:
            __import__(module_name)
            base_module = module_name
            break
        except ImportError:
            continue
    
    if not base_module:
        if RICH_AVAILABLE:
            console.print("   ❌ Cannot find FAIR-LLM module. Try: from fairlib import ...")
        else:
            print("   ❌ Cannot find FAIR-LLM module. Try: from fairlib import ...")
        return False
    
    components = [
        (f"{base_module}", "Main module"),
        (f"{base_module}.OpenAIAdapter", "OpenAI adapter"),
        (f"{base_module}.SimpleAgent", "Simple agent"),
        (f"{base_module}.HierarchicalAgentRunner", "Hierarchical runner"),
        (f"{base_module}.settings", "Settings module"),
        (f"{base_module}.utils.autograder_utils", "Autograder utilities"),
        (f"{base_module}.utils.document_processor", "Document processor"),
    ]
    
    all_good = True
    for import_path, description in components:
        try:
            if "." in import_path:
                parts = import_path.split(".")
                module = __import__(import_path.rsplit(".", 1)[0], fromlist=[parts[-1]])
                if len(parts) > 2:  # Sub-module import
                    getattr(module, parts[-1])
            else:
                __import__(import_path)
            
            if RICH_AVAILABLE:
                console.print(f"   ✅ {description}")
            else:
                print(f"   ✅ {description}")
        except ImportError as e:
            if RICH_AVAILABLE:
                console.print(f"   ❌ {description}: [red]{e}[/red]")
            else:
                print(f"   ❌ {description}: {e}")
            all_good = False
    
    return all_good

def suggest_fixes():
    """Suggest how to fix installation issues."""
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold yellow]💡 To fix installation issues:[/bold yellow]\n\n"
            "1. Make sure you're in a virtual environment:\n"
            "   [cyan]python -m venv venv\n"
            "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate[/cyan]\n\n"
            "2. Install all requirements:\n"
            "   [cyan]pip install -r requirements.txt[/cyan]\n\n"
            "3. If requirements.txt is missing:\n"
            "   [cyan]pip install fair-llm>=0.1 python-dotenv rich anthropic faiss-cpu seaborn pytest[/cyan]\n\n"
            "4. Verify Python version is 3.8+:\n"
            "   [cyan]python --version[/cyan]",
            style="yellow"
        ))
    else:
        print("\n💡 To fix installation issues:")
        print("   1. Make sure you're in a virtual environment:")
        print("      python -m venv venv")
        print("      source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("   2. Install all requirements:")
        print("      pip install -r requirements.txt")
        print("   3. Verify Python version is 3.8+:")
        print("      python --version")

def main():
    """Main verification function."""
    print_header()
    
    # Check for requirements.txt
    requirements = check_requirements_file()
    if not requirements:
        if RICH_AVAILABLE:
            console.print("[yellow]⚠️  No requirements.txt file found in current directory[/yellow]")
        else:
            print("⚠️  No requirements.txt file found in current directory")
    
    # Run checks
    py_ok, py_error = check_python_version()
    pkg_ok = check_package_installation(requirements)
    fairlib_ok = check_fairlib_components()
    
    # Summary
    if RICH_AVAILABLE:
        console.print("\n" + "=" * 60)
        if not py_ok:
            console.print(f"[bold red]❌ Python Version Issue: {py_error}[/bold red]")
            suggest_fixes()
        elif not pkg_ok or not fairlib_ok:
            console.print("[bold red]❌ Some packages are missing or not working[/bold red]")
            suggest_fixes()
        else:
            console.print("[bold green]✅ Perfect Installation![/bold green]")
            console.print("[bold cyan]🎉 Everything is configured and ready to use![/bold cyan]")
        
        # Next steps
        console.print("\n[bold cyan]📖 Next Steps:[/bold cyan]")
        console.print("   1. Explore the demos folder: [cyan]cd demos[/cyan]")
        console.print("   2. Run the essay autograder: [cyan]python demo_single_agent_calculator.py")
        console.print("   3. Run the code autograder: [cyan]python demo_rag_from_documents.py")
    else:
        print("\n" + "=" * 60)
        if not py_ok or not pkg_ok or not fairlib_ok:
            print("❌ Installation issues found")
            suggest_fixes()
        else:
            print("✅ Installation successful!")
            print("\n📖 Next Steps:")
            print("   1. Explore the demos folder")
            print("   2. Run the demo scripts in demos/")
    
    return py_ok and pkg_ok and fairlib_ok

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        if RICH_AVAILABLE:
            console.print(f"[bold red]❌ Unexpected error: {e}[/bold red]")
        else:
            print(f"❌ Unexpected error: {e}")
        sys.exit(1)