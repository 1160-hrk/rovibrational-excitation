#!/usr/bin/env python3
"""
テスト実行用スクリプト
"""
import subprocess
import sys
import os

def run_tests():
    """すべてのテストを実行"""
    # テストディレクトリに移動
    test_dir = os.path.dirname(__file__)
    os.chdir(test_dir)
    
    # Pytestでテスト実行
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "-v",  # verbose
            "--tb=short",  # short traceback
            "."
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_specific_test(test_name):
    """特定のテストを実行"""
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "-v",
            f"test_{test_name}.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running test {test_name}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        success = run_specific_test(test_name)
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1) 