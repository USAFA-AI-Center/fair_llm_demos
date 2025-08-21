"""
FAIR-LLM Installation Verification Script
Verifies that the fair-llm package and all dependencies are properly installed.
"""

import os
import sys
import subprocess
from typing import Tuple, Dict, Any
from pathlib import Path

# Try to import rich for better output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import print as rprint
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_status(status: bool, message: str, details: str = ""):
    """Print colored status message"""
    if RICH_AVAILABLE:
        emoji = "✅" if status else "❌"
        color = "green" if status else "red"
        console.print(f"{emoji} [{color}]{message}[/{color}]")
        if details:
            console.print(f"   {details}", style="dim")
    else:
        symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{symbol} {message}")
        if details:
            print(f"  {details}")


def print_warning(message: str):
    """Print warning message"""
    if RICH_AVAILABLE:
        console.print(f"⚠️  [yellow]{message}[/yellow]")
    else:
        print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def print_header(title: str):
    """Print section header"""
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"[bold blue]{title}[/bold blue]"))
    else:
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements (3.11+)"""
    current = sys.version_info[:2]
    required = (3, 11)
    
    status = current >= required
    message = f"Python {current[0]}.{current[1]}.{sys.version_info[2]}"
    
    if not status:
        message += f" (Required: >={required[0]}.{required[1]})"
    
    return status, message


def check_env_variable(var_name: str) -> Tuple[bool, str]:
    """Check if environment variable is set"""
    value = os.getenv(var_name)
    
    if value:
        # Mask the actual value for security
        if len(value) > 8:
            masked_value = value[:4] + "..." + value[-4:]
        else:
            masked_value = "***"
        return True, f"Set ({masked_value})"
    else:
        return False, "Not set"


def load_env_file():
    """Try to load .env file if it exists"""
    try:
        from dotenv import load_dotenv
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv()
            return True, "Loaded .env file"
        else:
            return False, ".env file not found (create from .env.example)"
    except ImportError:
        return False, "python-dotenv not installed"


def check_fairlib_import() -> Tuple[bool, str]:
    """Check if fairlib can be imported and test basic components"""
    try:
        # Try basic import
        import fairlib
        
        # Try to import core components
        components_to_test = [
            "settings",
            "OpenAIAdapter",
            "ToolRegistry",
            "SafeCalculatorTool",
            "WorkingMemory",
            "ReActPlanner",
            "SimpleAgent"
        ]
        
        failed_imports = []
        for component in components_to_test:
            try:
                getattr(fairlib, component)
            except AttributeError:
                failed_imports.append(component)
        
        if failed_imports:
            return False, f"Missing components: {', '.join(failed_imports)}"
        
        return True, "All core components available"
        
    except ImportError as e:
        return False, f"Import failed: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_package_installation(package_name: str) -> Dict[str, Any]:
    """Check if package is installed via pip"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            
            return {
                'installed': True,
                'version': info.get('Version', 'unknown'),
                'location': info.get('Location', 'unknown')
            }
        else:
            return {'installed': False, 'error': 'Package not found'}
    except Exception as e:
        return {'installed': False, 'error': str(e)}


def test_basic_functionality():
    """Test basic FAIR-LLM functionality"""
    print_header("Basic Functionality Test")
    
    try:
        from fairlib import SafeCalculatorTool, ToolRegistry
        
        # Test tool registration
        registry = ToolRegistry()
        calc_tool = SafeCalculatorTool()
        registry.register_tool(calc_tool)
        
        # Test tool listing
        tools = registry.get_all_tools()
        if "safe_calculator" in [t for t in tools]:
            print_status(True, "Tool registration works")
        else:
            print_status(False, "Tool registration failed")
            
        # Test tool execution (simple calculation)
        result = calc_tool.use("2 + 2")
        if "4" in str(result):
            print_status(True, "Tool execution works")
        else:
            print_status(False, f"Tool execution unexpected result: {result}")
            
    except Exception as e:
        print_status(False, f"Functionality test failed: {str(e)}")


def test_api_configuration():
    """Test if API keys are properly configured in fairlib.settings"""
    try:
        # Check if settings can access the environment variables
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if openai_key and anthropic_key:
            print_status(True, "API keys available to fairlib")
            return True
        else:
            print_status(False, "API keys not accessible to fairlib")
            return False
            
    except Exception as e:
        print_warning(f"Could not verify fairlib settings: {str(e)}")
        return False


def main():
    """Main verification function"""
    if RICH_AVAILABLE:
        console.print("[bold]🔍 FAIR-LLM Installation Verification[/bold]\n")
    else:
        print(f"{Colors.BOLD}FAIR-LLM Installation Verification{Colors.RESET}\n")
    
    all_checks_passed = True
    critical_failures = []
    
    # Check Python version
    print_header("Python Version Check")
    status, message = check_python_version()
    print_status(status, f"Python Version: {message}")
    if not status:
        all_checks_passed = False
        critical_failures.append("Python 3.11+ required")
    
    # Load environment variables from .env if it exists
    print_header("Environment Setup")
    env_status, env_message = load_env_file()
    if env_status:
        print_status(True, env_message)
    else:
        print_warning(env_message)
        print_warning("You can create a .env file from .env.example for convenience")
    
    # Check required environment variables
    print_header("API Keys (Environment Variables)")
    env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    env_var_status = True
    
    for var in env_vars:
        status, message = check_env_variable(var)
        print_status(status, f"{var}: {message}")
        if not status:
            env_var_status = False
            print_warning(f"  → Set in .env file or export {var}=your-key")
    
    if not env_var_status:
        all_checks_passed = False
        critical_failures.append("API keys not configured (set OPENAI_API_KEY and ANTHROPIC_API_KEY)")
    
    # Check GitHub token (optional, for installation help)
    gh_token_status, gh_token_message = check_env_variable("GH_TOKEN")
    if not gh_token_status:
        print_warning("GH_TOKEN: Not set (needed for installing private package)")
    
    # Check FAIR-LLM package installation
    print_header("Package Installation")
    
    # Check pip package
    pip_info = check_package_installation("fair-llm")
    if pip_info['installed']:
        print_status(True, f"fair-llm package: v{pip_info['version']}")
    else:
        print_status(False, f"fair-llm package: Not installed via pip")
        print_warning("  → Set GH_TOKEN: export GH_TOKEN=your-github-token")
        print_warning("  → Install with: pip install -r requirements.txt")
        all_checks_passed = False
        critical_failures.append("fair-llm package not installed")
    
    # Check fairlib import
    import_status, import_message = check_fairlib_import()
    print_status(import_status, f"fairlib import: {import_message}")
    if not import_status:
        all_checks_passed = False
        critical_failures.append("Cannot import fairlib components")
    
    # Test API configuration if fairlib imports successfully
    if import_status and env_var_status:
        print_header("API Configuration Test")
        test_api_configuration()
    
    # Test basic functionality if core checks pass
    if import_status:
        test_basic_functionality()
    
    # Check for demo files
    print_header("Demo Files")
    demo_files = [
        "demos/multi_agent_demo.py",
        "demos/demo_single_agent_calculator.py",
        "demos/demo_advanced_calculator_calculus.py",
        "demos/demo_rag_from_documents.py",
        "demos/demo_structured_output.py",
        "demos/demo_model_comparison.py",
    ]
    
    demos_found = []
    for demo in demo_files:
        if Path(demo).exists():
            demos_found.append(demo)
    
    if demos_found:
        print_status(True, f"Found {len(demos_found)} demo file(s)")
        for demo in demos_found[:3]:  # Show first 3
            print(f"  • {demo}")
        if len(demos_found) > 3:
            print(f"  • ... and {len(demos_found) - 3} more")
    else:
        print_warning("No demo files found in current directory")
    
    # Final summary
    print_header("Summary")
    
    if all_checks_passed:
        if RICH_AVAILABLE:
            console.print("[bold green]✅ All checks passed! Your environment is ready.[/bold green]")
            console.print("\n[cyan]Try running a demo:[/cyan]")
            console.print("  python multi_agent_demo.py")
        else:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ All checks passed! Your environment is ready.{Colors.RESET}")
            print("\nTry running a demo:")
            print("  python multi_agent_demo.py")
    else:
        if RICH_AVAILABLE:
            console.print("[bold red]❌ Some checks failed. Please fix the issues above.[/bold red]")
            if critical_failures:
                console.print("\n[red]Critical issues:[/red]")
                for issue in critical_failures:
                    console.print(f"  • {issue}")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ Some checks failed. Please fix the issues above.{Colors.RESET}")
            if critical_failures:
                print("\nCritical issues:")
                for issue in critical_failures:
                    print(f"  • {issue}")
        
        print("\n📚 Quick Fix Guide:")
        print("1. Set GitHub token: export GH_TOKEN=your_token")
        print("2. Install package: pip install -r requirements.txt")
        print("3. Set API keys:")
        print("   export OPENAI_API_KEY=your-key")
        print("   export ANTHROPIC_API_KEY=your-key")
        print("4. Run this script again: python verify_setup.py")
        
        sys.exit(1)


if __name__ == "__main__":
    main()