import os
import json
import pytest

REPORT_PATH = r"d:\trm_c2\TinyRecursiveModels\reports\arc_task_classification.json"

def test_file_exists_and_is_valid_json():
    assert os.path.exists(REPORT_PATH), f"Report file not found at {REPORT_PATH}"
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert isinstance(data, dict), "Root element should be a JSON object"

def test_root_keys():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    required_keys = {"taxonomy_schema", "summary", "tasks"}
    for key in required_keys:
        assert key in data, f"Root object is missing key: '{key}'"

def test_tasks_list_size():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    assert isinstance(tasks, list), "'tasks' must be a list"
    assert len(tasks) == 800, f"Expected exactly 800 tasks, but found {len(tasks)}"

def test_task_item_keys():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    required_keys = {
        "task_id", "ordinal", "identifier", "batch",
        "level1_family", "level2_sub_family", "level3_sub_sub_family"
    }
    for idx, task in enumerate(tasks):
        assert isinstance(task, dict), f"Task at index {idx} must be an object"
        for key in required_keys:
            assert key in task, f"Task at index {idx} is missing key: '{key}'"

def test_ordinals_sequential_and_range():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    ordinals = [task["ordinal"] for task in tasks]
    expected_ordinals = list(range(1, 801))
    assert ordinals == expected_ordinals, "Task ordinals are not sequential and running from 1 to 800"

def test_batch_assignment_by_ordinal():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    for idx, task in enumerate(tasks):
        ordinal = task["ordinal"]
        batch = task["batch"]
        if ordinal <= 400:
            assert batch == "Batch 1", f"Task with ordinal {ordinal} (<= 400) has batch '{batch}', expected 'Batch 1'"
        else:
            assert batch == "Batch 2", f"Task with ordinal {ordinal} (>= 401) has batch '{batch}', expected 'Batch 2'"

def test_summary_overall_counts():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    summary = data["summary"]
    
    assert "overall" in summary, "summary is missing 'overall' key"
    overall = summary["overall"]
    assert "total_tasks" in overall, "overall summary is missing 'total_tasks'"
    assert "batch_1_total" in overall, "overall summary is missing 'batch_1_total'"
    assert "batch_2_total" in overall, "overall summary is missing 'batch_2_total'"
    
    total_tasks = len(tasks)
    batch_1_count = sum(1 for t in tasks if t["batch"] == "Batch 1")
    batch_2_count = sum(1 for t in tasks if t["batch"] == "Batch 2")
    
    assert overall["total_tasks"] == total_tasks, f"Summary total_tasks ({overall['total_tasks']}) does not match actual task count ({total_tasks})"
    assert overall["batch_1_total"] == batch_1_count, f"Summary batch_1_total ({overall['batch_1_total']}) does not match actual Batch 1 task count ({batch_1_count})"
    assert overall["batch_2_total"] == batch_2_count, f"Summary batch_2_total ({overall['batch_2_total']}) does not match actual Batch 2 task count ({batch_2_count})"

def test_summary_by_category_counts():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    summary = data["summary"]
    
    assert "by_category" in summary, "summary is missing 'by_category' key"
    by_category = summary["by_category"]
    assert isinstance(by_category, list), "summary['by_category'] must be a list"
    
    # Track which categories exist in the tasks
    task_counts = {}
    for task in tasks:
        key = (task["level1_family"], task["level2_sub_family"], task["level3_sub_sub_family"])
        batch = task["batch"]
        if key not in task_counts:
            task_counts[key] = {"count": 0, "Batch 1": 0, "Batch 2": 0}
        task_counts[key]["count"] += 1
        task_counts[key][batch] += 1
        
    # Check that each category in summary by_category has matching counts
    visited_categories = set()
    for cat in by_category:
        level1 = cat.get("level1_family")
        level2 = cat.get("level2_sub_family")
        level3 = cat.get("level3_sub_sub_family")
        key = (level1, level2, level3)
        visited_categories.add(key)
        
        expected = task_counts.get(key, {"count": 0, "Batch 1": 0, "Batch 2": 0})
        
        assert cat.get("count") == expected["count"], f"Category {key} has count {cat.get('count')}, expected {expected['count']}"
        assert cat.get("batch_1_count") == expected["Batch 1"], f"Category {key} has batch_1_count {cat.get('batch_1_count')}, expected {expected['Batch 1']}"
        assert cat.get("batch_2_count") == expected["Batch 2"], f"Category {key} has batch_2_count {cat.get('batch_2_count')}, expected {expected['Batch 2']}"
        
    # Check that there are no categories present in the tasks that are not defined in summary by_category
    for key, counts in task_counts.items():
        if counts["count"] > 0:
            assert key in visited_categories, f"Category {key} has tasks assigned but is missing from summary 'by_category'"

def test_all_tasks_mapped_to_schema_categories():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    schema = data["taxonomy_schema"]
    tasks = data["tasks"]
    for idx, t in enumerate(tasks):
        l1 = t["level1_family"]
        l2 = t["level2_sub_family"]
        l3 = t["level3_sub_sub_family"]
        
        assert l1 in schema, f"Task {t['task_id']} (idx {idx}) has Level 1 family '{l1}' not in taxonomy_schema"
        assert l2 in schema[l1].get("sub_families", {}), f"Task {t['task_id']} (idx {idx}) has Level 2 family '{l2}' not under '{l1}' in taxonomy_schema"
        assert l3 in schema[l1]["sub_families"][l2].get("sub_sub_families", {}), f"Task {t['task_id']} (idx {idx}) has Level 3 family '{l3}' not under '{l2}' in taxonomy_schema"

def test_task_id_format_and_uniqueness():
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    tasks = data["tasks"]
    task_ids = set()
    identifiers = set()
    for idx, t in enumerate(tasks):
        tid = t["task_id"]
        ord_val = t["ordinal"]
        ident = t["identifier"]
        
        # Check uniqueness
        assert tid not in task_ids, f"Duplicate task_id found: '{tid}'"
        assert ident not in identifiers, f"Duplicate identifier found: '{ident}'"
        task_ids.add(tid)
        identifiers.add(ident)
        
        # Check format
        expected_tid = f"{ord_val:04d}_{ident}"
        assert tid == expected_tid, f"Task at index {idx} has mismatched task_id '{tid}', expected '{expected_tid}'"


