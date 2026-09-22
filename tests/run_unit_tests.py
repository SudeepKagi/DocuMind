"""
Runner for unit tests in tests/ directory.
Executes test classes directly without requiring external pytest runner.
"""

import sys
import inspect
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "backend" / "packages"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import test_calculation
import test_retrieval_isolation
import test_extraction
import test_generation_termination
import test_context_gate

modules = [
    test_calculation,
    test_retrieval_isolation,
    test_extraction,
    test_generation_termination,
    test_context_gate,
]

total_passed = 0
total_failed = 0
errors = []

print("=" * 60)
print("DocuMind W: Running Consolidated Architecture Unit Tests")
print("=" * 60)

for mod in modules:
    print(f"\n--- Running tests in {mod.__name__} ---")
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and name.startswith("Test"):
            inst = obj()
            for m_name, m_func in inspect.getmembers(inst, predicate=inspect.ismethod):
                if m_name.startswith("test_"):
                    try:
                        m_func()
                        print(f"  [PASS] {name}.{m_name}")
                        total_passed += 1
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print(f"  [FAIL] {name}.{m_name}: {e}")
                        total_failed += 1
                        errors.append((f"{mod.__name__}.{name}.{m_name}", tb))

print("\n" + "=" * 60)
print(f"SUMMARY: {total_passed} PASSED, {total_failed} FAILED")
print("=" * 60)

if total_failed > 0:
    print("\nFailures:")
    for test_id, err in errors:
        print(f"  - {test_id}: {err}")
    sys.exit(1)
else:
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
