#!/usr/bin/env python3
"""
Package integrity check script.
Run this before uploading to GitHub to ensure everything is working.
"""

import ast
import os
import sys
from pathlib import Path

def check_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def find_python_files(directory):
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def check_imports(file_path):
    """Check for common import issues."""
    issues = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for relative imports that might be problematic
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            line = line.strip()
            # Skip __future__ imports and comments
            if line.startswith('from __future__') or line.startswith('#'):
                continue
            if line.startswith('from _') and 'import' in line:
                issues.append(f"Line {i}: Potential relative import issue: {line}")
            if 'import _' in line and not line.startswith('#'):
                issues.append(f"Line {i}: Potential underscore import issue: {line}")
                
    except Exception as e:
        issues.append(f"Error reading file: {str(e)}")
        
    return issues

def main():
    """Run package integrity checks."""
    print("🔍 IRVI Package Integrity Check")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('irvi'):
        print("❌ Error: 'irvi' package directory not found!")
        print("   Please run this script from the package root directory.")
        sys.exit(1)
    
    # Find all Python files
    python_files = find_python_files('irvi')
    test_files = find_python_files('tests') if os.path.exists('tests') else []
    all_files = python_files + test_files
    
    print(f"📁 Found {len(python_files)} Python files in package")
    if test_files:
        print(f"🧪 Found {len(test_files)} test files")
    
    # Check syntax
    print("\n📝 Checking Python syntax...")
    syntax_errors = 0
    for file_path in all_files:
        valid, error = check_syntax(file_path)
        if not valid:
            print(f"❌ Syntax error in {file_path}: {error}")
            syntax_errors += 1
        else:
            print(f"✅ {file_path}")
    
    # Check imports
    print("\n🔗 Checking import statements...")
    import_issues = 0
    for file_path in all_files:
        issues = check_imports(file_path)
        if issues:
            print(f"⚠️  Issues in {file_path}:")
            for issue in issues:
                print(f"   {issue}")
            import_issues += len(issues)
    
    # Check required files
    print("\n📄 Checking required files...")
    required_files = [
        'README.md',
        'LICENSE',
        'pyproject.toml',
        'irvi/__init__.py',
        'irvi/model/__init__.py',
        'irvi/module/__init__.py',
        'irvi/data/__init__.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_files.append(file_path)
    
    # Check package structure
    print("\n🏗️  Package structure:")
    expected_structure = {
        'irvi': ['__init__.py'],
        'irvi/model': ['__init__.py', '_irvi.py'],
        'irvi/module': ['__init__.py', '_irvae.py'],
        'irvi/data': ['__init__.py', '_tcr_field.py'],
    }
    
    structure_issues = 0
    for directory, expected_files in expected_structure.items():
        if os.path.exists(directory):
            print(f"✅ {directory}/")
            for file in expected_files:
                file_path = os.path.join(directory, file)
                if os.path.exists(file_path):
                    print(f"  ✅ {file}")
                else:
                    print(f"  ❌ Missing: {file}")
                    structure_issues += 1
        else:
            print(f"❌ Missing directory: {directory}")
            structure_issues += 1
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 SUMMARY")
    print("=" * 40)
    
    total_issues = syntax_errors + import_issues + len(missing_files) + structure_issues
    
    if total_issues == 0:
        print("🎉 All checks passed! Package is ready for GitHub.")
        print("\n📤 Next steps:")
        print("   1. git add .")
        print("   2. git commit -m 'Initial package release'")
        print("   3. git push origin main")
        sys.exit(0)
    else:
        print(f"⚠️  Found {total_issues} issue(s) that should be addressed:")
        if syntax_errors > 0:
            print(f"   - {syntax_errors} syntax error(s)")
        if import_issues > 0:
            print(f"   - {import_issues} import issue(s)")
        if missing_files:
            print(f"   - {len(missing_files)} missing file(s)")
        if structure_issues > 0:
            print(f"   - {structure_issues} structure issue(s)")
        sys.exit(1)

if __name__ == "__main__":
    main()
