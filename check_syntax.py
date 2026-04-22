import ast
import sys

try:
    with open('streamlit_app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✓ File is syntactically valid")
except SyntaxError as e:
    print(f"✗ Syntax Error on line {e.lineno}")
    print(f"  Text: {repr(e.text)}")
    print(f"  Msg: {e.msg}")
    if e.text:
        print(f"  {' ' * (e.offset - 1) if e.offset else ''}^")
