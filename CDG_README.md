# Cognitive Decision Graph (CDG)

A data structure for representing Python programs as decision trees, abstracting away syntactic differences to capture cognitive/semantic equivalence.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Data Structures](#data-structures)
   - [Interval Arithmetic](#interval-arithmetic)
   - [CDG Node Types](#cdg-node-types)
4. [CDG Builder](#cdg-builder)
5. [Similarity Computation](#similarity-computation)
6. [Usage Examples](#usage-examples)

---

## Overview

The CDG framework transforms Python source code into a semantic representation that captures the **cognitive structure** of a program—its decision logic, input/output behavior, and control flow—independent of syntactic variations.

### Key Benefits

- **Syntactic Invariance**: Recognizes that `if x > 1.6 and x < 1.9` and `if 1.6 < x < 1.9` are semantically equivalent
- **Order Independence**: Branches in different orders (if/elif/else) are recognized as equivalent if they cover the same regions
- **Constant Resolution**: Automatically resolves named constants like `HEIGHT_MIN = 1.6` to their numeric values
- **Multi-line Input Handling**: Correctly tracks input patterns across multiple lines

---

## Core Concepts

### Decision Regions

The fundamental abstraction in CDG is the **Region**—a set of numeric intervals that define where a condition is true.

For example:
- `height < 1.6` → Region: `(-∞, 1.6)`
- `height >= 1.9` → Region: `[1.9, ∞)`
- `1.6 < height < 1.9` → Region: `(1.6, 1.9)`
- `height != 1.7` → Region: `(-∞, 1.7) ∪ (1.7, ∞)`

### Semantic Equivalence

Two programs are semantically equivalent in CDG terms if they:
1. Accept the same types of input
2. Have the same decision regions
3. Produce the same outputs for the same inputs

---

## Data Structures

### Interval Arithmetic

#### `Interval`

Represents a numeric interval with inclusive/exclusive bounds.

```python
class Interval:
    def __init__(
        self,
        lower: float = float('-inf'),
        upper: float = float('inf'),
        lower_inclusive: bool = False,
        upper_inclusive: bool = False
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lower` | `float` | `-inf` | Lower bound of the interval |
| `upper` | `float` | `inf` | Upper bound of the interval |
| `lower_inclusive` | `bool` | `False` | Whether the lower bound is included (`[` vs `(`) |
| `upper_inclusive` | `bool` | `False` | Whether the upper bound is included (`]` vs `)`) |

**Notation:**
- `(-∞, 1.6)` - All values less than 1.6
- `[1.6, 1.9]` - All values from 1.6 to 1.9 inclusive
- `{1.6}` - Point interval (single value)

**Key Methods:**
- `Interval.point(value)` - Create a point interval `{value}`
- `Interval.from_comparison(op, value)` - Create from comparison like `'<', 1.6`
- `contains(value)` - Check if value is in the interval
- `intersect(other)` - Return intersection of two intervals
- `is_empty()` - Check if interval is empty

---

#### `Region`

A union of intervals representing a decision region.

```python
class Region:
    def __init__(self, intervals: List[Interval] = None)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `intervals` | `List[Interval]` | `[]` | List of intervals in the region |

**Key Methods:**
- `Region.full()` - Create region covering all real numbers `(-∞, ∞)`
- `Region.empty()` - Create empty region `∅`
- `Region.from_constraint(op, value)` - Create from constraint like `'>=', 1.9`
- `intersect(other)` - AND logic: intersection of two regions
- `union(other)` - OR logic: union of two regions
- `complement()` - NOT logic: everything not in this region

**Example:**
```python
# Create region for: height > 1.6 AND height < 1.9
region_gt = Region.from_constraint('>', 1.6)   # (1.6, ∞)
region_lt = Region.from_constraint('<', 1.9)   # (-∞, 1.9)
result = region_gt.intersect(region_lt)        # (1.6, 1.9)
```

---

### CDG Node Types

All nodes inherit from `CDGNode` base class:

```python
class CDGNode(ABC):
    children: List[CDGNode]    # Child nodes
    parent: Optional[CDGNode]  # Parent node
    
    def add_child(child: CDGNode)
    def semantic_signature() -> Any  # For comparison
```

---

#### `BodyNode`

Root node representing the program body.

```python
class BodyNode(CDGNode):
    def __init__(self, name: str = "main")
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"main"` | Name of the function/program |

**Children:** Any CDG nodes (InputNode, ControlNode, ActionNode)

---

#### `InputNode`

Represents user input capture.

```python
class InputNode(CDGNode):
    def __init__(
        self,
        var_name: str,
        input_type: Optional[str] = None,
        prompt: str = ""
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `var_name` | `str` | required | Variable name the input is assigned to |
| `input_type` | `str` | `None` | Type conversion: `'float'`, `'int'`, `'str'`, or `None` |
| `prompt` | `str` | `""` | Prompt text shown to user |

**Recognized Patterns:**
```python
# Single-line patterns
height = float(input("Enter height: "))  # type='float'
age = int(input("Enter age: "))          # type='int'
name = input("Enter name: ")             # type='str'

# Multi-line patterns (also recognized)
height_str = input("Enter height: ")
height = float(height_str)               # Linked as type='float'
```

**Semantic Signature:** `('input', input_type)` - Variable names are ignored for comparison.

---

#### `ControlNode`

Represents control flow structures (if/while/for).

```python
class ControlNode(CDGNode):
    def __init__(
        self,
        control_type: ControlType,
        var_name: Optional[str] = None
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `control_type` | `ControlType` | required | Type of control: `IF`, `WHILE`, `FOR` |
| `var_name` | `str` | `None` | The decision variable being tested |

**Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `branches` | `List[BranchNode]` | List of branches for if/elif/else chains |

**ControlType Enum:**
```python
class ControlType(Enum):
    IF = "if"
    WHILE = "while"
    FOR = "for"
```

**Semantic Signature:** Uses `frozenset` of branch signatures for **order-independent** comparison.

---

#### `BranchNode`

Represents a branch within a control structure.

```python
class BranchNode(CDGNode):
    def __init__(
        self,
        region: Region,
        is_else: bool = False
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `Region` | required | The numeric region where this branch executes |
| `is_else` | `bool` | `False` | Whether this is an else branch |

**Children:** ActionNodes and nested ControlNodes

**Semantic Signature:** `('branch', str(region), child_signatures)`

---

#### `ActionNode`

Represents observable behavior (print, return, etc.).

```python
class ActionNode(CDGNode):
    def __init__(
        self,
        action_type: ActionType,
        message: str = "",
        value: Any = None
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_type` | `ActionType` | required | Type of action |
| `message` | `str` | `""` | For print: the output message |
| `value` | `Any` | `None` | For return/assign: the value |

**ActionType Enum:**
```python
class ActionType(Enum):
    PRINT = "print"
    RETURN = "return"
    ASSIGN = "assign"
    CALL = "call"
    BREAK = "break"
    CONTINUE = "continue"
```

**Semantic Signature:**
- Print: `('action', 'print', message)`
- Return: `('action', 'return', str(value))`
- Others: `('action', action_type)`

---

#### `RegionNode`

Represents a numeric decision region (standalone, not within branch).

```python
class RegionNode(CDGNode):
    def __init__(
        self,
        region: Region,
        var_name: str = ""
    )
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `region` | `Region` | required | The decision region |
| `var_name` | `str` | `""` | Variable name (metadata only) |

---

## CDG Builder

### `CDGBuilder`

Builds a Cognitive Decision Graph from Python source code.

```python
class CDGBuilder:
    def __init__(self, code: str)
```

**Internal State:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `var_types` | `Dict[str, str]` | Maps variable names to their types |
| `input_vars` | `Dict[str, str]` | Tracks raw input() for multi-line patterns |
| `constants` | `Dict[str, float]` | Named constants resolved to numeric values |

### Features

#### 1. Constant Resolution

Constants defined at module level or inside functions are automatically resolved:

```python
HEIGHT_MINIMUM = 1.6
HEIGHT_MAXIMUM = 1.9

def main():
    if height > HEIGHT_MINIMUM:  # Resolved to 1.6
        ...
```

**Recognition Rules:**
- Variable name must be UPPERCASE (e.g., `HEIGHT_MIN`, `MAX_VALUE`)
- Assigned to a numeric literal (int or float)
- Supports negative numbers

#### 2. Multi-line Input Tracking

Handles input patterns split across lines:

```python
# These are equivalent:
height = float(input("Enter height: "))

# And:
height_str = input("Enter height: ")
height = float(height_str)
```

#### 3. Condition to Region Conversion

Automatically converts Python conditions to regions:

| Python Condition | Region |
|-----------------|--------|
| `x < 1.6` | `(-∞, 1.6)` |
| `x <= 1.6` | `(-∞, 1.6]` |
| `x > 1.6` | `(1.6, ∞)` |
| `x >= 1.6` | `[1.6, ∞)` |
| `x == 1.6` | `{1.6}` |
| `x != 1.6` | `(-∞, 1.6) ∪ (1.6, ∞)` |
| `x > 1.6 and x < 1.9` | `(1.6, 1.9)` |
| `1.6 < x < 1.9` | `(1.6, 1.9)` |
| `x < 1.6 or x > 1.9` | `(-∞, 1.6) ∪ (1.9, ∞)` |

---

## Similarity Computation

### Overview

The `cdg_similarity()` function compares two CDGs and returns multiple similarity scores:

```python
def cdg_similarity(cdg1: BodyNode, cdg2: BodyNode) -> Dict[str, float]
```

### Similarity Metrics

| Metric | Description | What It Compares |
|--------|-------------|------------------|
| `structural` | Control flow structure similarity | Control types, branch counts, regions (NOT action content) |
| `behavioral` | Input/output behavior similarity | Weighted: 30% input + 70% action |
| `regional` | Decision region similarity | Set of all regions (Jaccard) |
| `input` | Input similarity | Input types (ignores variable names/prompts) |
| `action` | Action similarity | Print messages, return values |
| `overall` | Weighted combination | 20% structural + 30% behavioral + 50% regional |

### Computation Details

#### Structural Similarity

Compares the **shape** of the decision tree without considering action content:

```
structural_signature includes:
- Input types (but not prompts)
- Control node types and branch counts
- Branch regions (but not branch content)
- Action types (but not messages/values)
```

If exact signatures match → `1.0`
Otherwise → weighted combination:
- 30% control node similarity
- 50% region similarity  
- 20% input type similarity

#### Behavioral Similarity

```
behavioral = 0.3 × input_similarity + 0.7 × action_similarity
```

- **Input similarity**: Jaccard of input type signatures
- **Action similarity**: Jaccard of full action signatures (including messages)

#### Regional Similarity

```
regional = Jaccard(regions_1, regions_2)
```

Uses string representation of regions for set comparison.

#### Overall Similarity

```
overall = 0.2 × structural + 0.3 × behavioral + 0.5 × regional
```

### Jaccard Similarity

Used for all set comparisons:

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

- Returns `1.0` if both sets are empty
- Returns `0.0` if one set is empty and other is not

---

## Usage Examples

### Basic Usage

```python
from cdg import build_cdg, compare_programs, print_cdg

# Build and print a CDG
code = '''
def main():
    height = float(input("Enter height: "))
    if height < 1.6:
        print("Too short")
    elif height > 1.9:
        print("Too tall")
    else:
        print("Just right")
'''
print_cdg(code)
```

**Output:**
```
BodyNode(main)
  InputNode(type=float, prompt="Enter height: ")
  ControlNode(type=if, branches=3)
    BranchNode(if/elif, region=(-∞, 1.6))
      ActionNode(print, msg="Too short")
    BranchNode(if/elif, region=(1.9, ∞))
      ActionNode(print, msg="Too tall")
    BranchNode(else, region=[1.6, 1.9])
      ActionNode(print, msg="Just right")
```

### Comparing Programs

```python
code_a = '''
def main():
    h = float(input("Height: "))
    if h > 1.6 and h < 1.9:
        print("OK")
    else:
        print("Not OK")
'''

code_b = '''
def main():
    height = float(input("Height: "))
    if 1.6 < height < 1.9:
        print("OK")
    else:
        print("Not OK")
'''

similarity = compare_programs(code_a, code_b)
# Result: All metrics = 1.0 (semantically identical)
```

### Working with CDG Objects

```python
from cdg import CDGBuilder, ControlNode, BranchNode

builder = CDGBuilder(code)
cdg = builder.get_cdg()

# Traverse the tree
for child in cdg.children:
    print(child)
    if isinstance(child, ControlNode):
        for branch in child.branches:
            print(f"  Region: {branch.region}")
```

### Creating Regions Manually

```python
from cdg import Region, Interval

# From constraints
region1 = Region.from_constraint('>', 1.6)   # (1.6, ∞)
region2 = Region.from_constraint('<', 1.9)   # (-∞, 1.9)

# Intersection (AND)
both = region1.intersect(region2)            # (1.6, 1.9)

# Union (OR)
either = region1.union(region2)              # (-∞, ∞)

# Complement (NOT)
outside = both.complement()                  # (-∞, 1.6] ∪ [1.9, ∞)
```

---

## CDG Tree Structure

```
BodyNode("main")
├── InputNode(type=float, prompt="...")
└── ControlNode(type=if)
    ├── BranchNode(region=(-∞, 1.6))
    │   └── ActionNode(print, "Below minimum")
    ├── BranchNode(region=(1.6, 1.9))
    │   └── ActionNode(print, "Correct height")
    └── BranchNode(region=[1.9, ∞))
        └── ActionNode(print, "Above maximum")
```

---

## Interpretation Guide

### When Structural = 1.0 but Action < 1.0

The programs have the same decision tree shape but different output messages (possibly typos).

### When Regional = 1.0 but Structural < 1.0

Same decision regions but different control flow (e.g., one uses `while True`, the other doesn't).

### When All = 1.0

Programs are semantically equivalent—they will produce the same outputs for the same inputs.

