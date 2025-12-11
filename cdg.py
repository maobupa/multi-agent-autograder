"""
Cognitive Decision Graph (CDG)
==============================
A data structure for representing Python programs as decision trees,
abstracting away syntactic differences to capture cognitive/semantic equivalence.

Node Types:
- InputNode: User input capture (type, prompt)
- ControlNode: Flow control structures (if/while/for)
- RegionNode: Numeric decision regions (the core abstraction)
- ActionNode: Observable behaviors (print, return, etc.)
"""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple, Set, Dict, Any
from enum import Enum
import math


# =============================================================================
# INTERVAL ARITHMETIC
# =============================================================================

class Interval:
    """
    Represents a numeric interval with inclusive/exclusive bounds.
    Supports: (-inf, 1.6), [1.6, 1.9], {1.6} (point), etc.
    """
    
    def __init__(
        self,
        lower: float = float('-inf'),
        upper: float = float('inf'),
        lower_inclusive: bool = False,
        upper_inclusive: bool = False
    ):
        self.lower = lower
        self.upper = upper
        self.lower_inclusive = lower_inclusive
        self.upper_inclusive = upper_inclusive
        
        # Infinity bounds are never inclusive
        if math.isinf(self.lower):
            self.lower_inclusive = False
        if math.isinf(self.upper):
            self.upper_inclusive = False
    
    @classmethod
    def point(cls, value: float) -> 'Interval':
        """Create a point interval {value}"""
        return cls(value, value, True, True)
    
    @classmethod
    def from_comparison(cls, op: str, value: float) -> 'Interval':
        """Create interval from a comparison like '< 1.6' or '>= 1.9'"""
        if op == '<':
            return cls(float('-inf'), value, False, False)
        elif op == '<=':
            return cls(float('-inf'), value, False, True)
        elif op == '>':
            return cls(value, float('inf'), False, False)
        elif op == '>=':
            return cls(value, float('inf'), True, False)
        elif op == '==':
            return cls.point(value)
        elif op == '!=':
            # Returns None - caller should handle as complement
            return None
        else:
            raise ValueError(f"Unknown operator: {op}")
    
    def contains(self, value: float) -> bool:
        """Check if value is in this interval"""
        if self.lower == self.upper:  # Point interval
            return self.lower_inclusive and value == self.lower
        
        lower_ok = (value > self.lower) or (self.lower_inclusive and value == self.lower)
        upper_ok = (value < self.upper) or (self.upper_inclusive and value == self.upper)
        return lower_ok and upper_ok
    
    def intersect(self, other: 'Interval') -> Optional['Interval']:
        """Return intersection of two intervals, or None if empty"""
        # New lower bound
        if self.lower > other.lower:
            new_lower = self.lower
            new_lower_inc = self.lower_inclusive
        elif self.lower < other.lower:
            new_lower = other.lower
            new_lower_inc = other.lower_inclusive
        else:  # Equal
            new_lower = self.lower
            new_lower_inc = self.lower_inclusive and other.lower_inclusive
        
        # New upper bound
        if self.upper < other.upper:
            new_upper = self.upper
            new_upper_inc = self.upper_inclusive
        elif self.upper > other.upper:
            new_upper = other.upper
            new_upper_inc = other.upper_inclusive
        else:  # Equal
            new_upper = self.upper
            new_upper_inc = self.upper_inclusive and other.upper_inclusive
        
        # Check if valid
        if new_lower > new_upper:
            return None
        if new_lower == new_upper and not (new_lower_inc and new_upper_inc):
            return None
        
        return Interval(new_lower, new_upper, new_lower_inc, new_upper_inc)
    
    def is_empty(self) -> bool:
        """Check if interval is empty"""
        if self.lower > self.upper:
            return True
        if self.lower == self.upper:
            return not (self.lower_inclusive and self.upper_inclusive)
        return False
    
    def __eq__(self, other: 'Interval') -> bool:
        if not isinstance(other, Interval):
            return False
        return (self.lower == other.lower and 
                self.upper == other.upper and
                self.lower_inclusive == other.lower_inclusive and
                self.upper_inclusive == other.upper_inclusive)
    
    def __hash__(self):
        return hash((self.lower, self.upper, self.lower_inclusive, self.upper_inclusive))
    
    def __repr__(self) -> str:
        left = '[' if self.lower_inclusive else '('
        right = ']' if self.upper_inclusive else ')'
        
        lower_str = '-∞' if math.isinf(self.lower) and self.lower < 0 else str(self.lower)
        upper_str = '∞' if math.isinf(self.upper) and self.upper > 0 else str(self.upper)
        
        # Point notation for single values
        if self.lower == self.upper and self.lower_inclusive:
            return f"{{{self.lower}}}"
        
        return f"{left}{lower_str}, {upper_str}{right}"


class Region:
    """
    A union of intervals representing a decision region.
    Handles complex conditions like: x != 1.7 → (-∞, 1.7) ∪ (1.7, ∞)
    """
    
    def __init__(self, intervals: List[Interval] = None):
        self.intervals: List[Interval] = intervals or []
        self._normalize()
    
    def _normalize(self):
        """Sort and merge overlapping intervals"""
        if not self.intervals:
            return
        
        # Remove empty intervals
        self.intervals = [i for i in self.intervals if not i.is_empty()]
        
        if not self.intervals:
            return
        
        # Sort by lower bound
        self.intervals.sort(key=lambda i: (i.lower, not i.lower_inclusive))
        
        # Merge overlapping intervals
        merged = [self.intervals[0]]
        for current in self.intervals[1:]:
            last = merged[-1]
            
            # Check if they overlap or touch
            if (current.lower < last.upper or 
                (current.lower == last.upper and (current.lower_inclusive or last.upper_inclusive))):
                # Merge
                new_upper = max(last.upper, current.upper)
                new_upper_inc = (
                    (last.upper_inclusive if last.upper >= current.upper else current.upper_inclusive)
                    if last.upper != current.upper else
                    (last.upper_inclusive or current.upper_inclusive)
                )
                merged[-1] = Interval(last.lower, new_upper, last.lower_inclusive, new_upper_inc)
            else:
                merged.append(current)
        
        self.intervals = merged
    
    @classmethod
    def full(cls) -> 'Region':
        """Create region covering all real numbers"""
        return cls([Interval()])
    
    @classmethod
    def empty(cls) -> 'Region':
        """Create empty region"""
        return cls([])
    
    @classmethod
    def from_constraint(cls, op: str, value: float) -> 'Region':
        """Create region from single constraint"""
        if op == '!=':
            # Special case: complement of a point
            return cls([
                Interval(float('-inf'), value, False, False),
                Interval(value, float('inf'), False, False)
            ])
        
        interval = Interval.from_comparison(op, value)
        return cls([interval]) if interval else cls.empty()
    
    def intersect(self, other: 'Region') -> 'Region':
        """Intersection of two regions (AND logic)"""
        result_intervals = []
        for i1 in self.intervals:
            for i2 in other.intervals:
                intersection = i1.intersect(i2)
                if intersection and not intersection.is_empty():
                    result_intervals.append(intersection)
        return Region(result_intervals)
    
    def union(self, other: 'Region') -> 'Region':
        """Union of two regions (OR logic)"""
        return Region(self.intervals + other.intervals)
    
    def complement(self, domain: 'Region' = None) -> 'Region':
        """
        Complement of this region within domain (default: all reals).
        Used for computing 'else' branches.
        """
        if domain is None:
            domain = Region.full()
        
        if not self.intervals:
            return domain
        
        # Build complement by subtracting each interval from domain
        result = domain
        for interval in self.intervals:
            result = result._subtract_interval(interval)
        
        return result
    
    def _subtract_interval(self, to_remove: Interval) -> 'Region':
        """Remove an interval from this region"""
        result_intervals = []
        
        for interval in self.intervals:
            # Case 1: No overlap - keep interval
            if interval.upper < to_remove.lower or interval.lower > to_remove.upper:
                result_intervals.append(interval)
                continue
            
            # Case 2: to_remove completely covers interval - remove it
            if (to_remove.lower <= interval.lower and to_remove.upper >= interval.upper):
                # Check boundary conditions
                if to_remove.lower == interval.lower and not to_remove.lower_inclusive and interval.lower_inclusive:
                    result_intervals.append(Interval.point(interval.lower))
                if to_remove.upper == interval.upper and not to_remove.upper_inclusive and interval.upper_inclusive:
                    result_intervals.append(Interval.point(interval.upper))
                continue
            
            # Case 3: Partial overlap - split
            # Left part
            if interval.lower < to_remove.lower:
                result_intervals.append(Interval(
                    interval.lower, to_remove.lower,
                    interval.lower_inclusive, not to_remove.lower_inclusive
                ))
            elif interval.lower == to_remove.lower and interval.lower_inclusive and not to_remove.lower_inclusive:
                result_intervals.append(Interval.point(interval.lower))
            
            # Right part
            if interval.upper > to_remove.upper:
                result_intervals.append(Interval(
                    to_remove.upper, interval.upper,
                    not to_remove.upper_inclusive, interval.upper_inclusive
                ))
            elif interval.upper == to_remove.upper and interval.upper_inclusive and not to_remove.upper_inclusive:
                result_intervals.append(Interval.point(interval.upper))
        
        return Region(result_intervals)
    
    def is_empty(self) -> bool:
        return len(self.intervals) == 0
    
    def __eq__(self, other: 'Region') -> bool:
        if not isinstance(other, Region):
            return False
        return self.intervals == other.intervals
    
    def __hash__(self):
        return hash(tuple(self.intervals))
    
    def __repr__(self) -> str:
        if not self.intervals:
            return "∅"
        return " ∪ ".join(str(i) for i in self.intervals)


# =============================================================================
# CDG NODE TYPES
# =============================================================================

class CDGNode(ABC):
    """Base class for all CDG nodes"""
    
    def __init__(self):
        self.children: List['CDGNode'] = []
        self.parent: Optional['CDGNode'] = None
    
    def add_child(self, child: 'CDGNode'):
        child.parent = self
        self.children.append(child)
    
    @abstractmethod
    def semantic_signature(self) -> Any:
        """Return a hashable signature for semantic comparison"""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


class InputNode(CDGNode):
    """
    Represents user input capture.
    
    Attributes:
        var_name: Variable name the input is assigned to
        input_type: Type conversion applied (float, int, str, None)
        prompt: Prompt text shown to user
    """
    
    def __init__(self, var_name: str, input_type: Optional[str] = None, prompt: str = ""):
        super().__init__()
        self.var_name = var_name
        self.input_type = input_type  # 'float', 'int', 'str', or None
        self.prompt = prompt
    
    def semantic_signature(self) -> Tuple:
        # For semantic comparison, variable name doesn't matter
        # Only the type and that we're capturing input
        return ('input', self.input_type)
    
    def __repr__(self) -> str:
        type_str = f"type={self.input_type}" if self.input_type else "type=str"
        prompt_str = f'prompt="{self.prompt}"' if self.prompt else ""
        return f"InputNode({type_str}, {prompt_str})"


class ControlType(Enum):
    IF = "if"
    WHILE = "while"
    FOR = "for"


class ControlNode(CDGNode):
    """
    Represents control flow structure (if/while/for).
    
    Does NOT capture predicate details - those go in RegionNodes.
    Only captures structural metadata.
    """
    
    def __init__(self, control_type: ControlType, var_name: Optional[str] = None):
        super().__init__()
        self.control_type = control_type
        self.var_name = var_name  # The variable being tested (for semantic linking)
        self.branches: List['BranchNode'] = []  # For if/elif/else chains
    
    def add_branch(self, branch: 'BranchNode'):
        branch.parent = self
        self.branches.append(branch)
    
    def semantic_signature(self) -> Tuple:
        # Use frozenset for branch signatures to make comparison order-independent
        # This ensures that if/elif branches in different orders are semantically equivalent
        branch_sigs = frozenset(b.semantic_signature() for b in self.branches)
        return ('control', self.control_type.value, branch_sigs)
    
    def __repr__(self) -> str:
        return f"ControlNode(type={self.control_type.value}, branches={len(self.branches)})"


class BranchNode(CDGNode):
    """
    Represents a branch within a control structure.
    Contains the region and actions for that branch.
    """
    
    def __init__(self, region: Region, is_else: bool = False):
        super().__init__()
        self.region = region
        self.is_else = is_else
    
    def semantic_signature(self) -> Tuple:
        # Region-based signature
        child_sigs = tuple(c.semantic_signature() for c in self.children)
        return ('branch', str(self.region), child_sigs)
    
    def __repr__(self) -> str:
        branch_type = "else" if self.is_else else "if/elif"
        return f"BranchNode({branch_type}, region={self.region})"


class RegionNode(CDGNode):
    """
    Represents a numeric decision region - THE CORE ABSTRACTION.
    
    Captures the semantic region regardless of how it was expressed:
    - height < 1.6
    - 1.6 > height  
    - height <= 1.6 - 0.0001
    
    All become: region = (-∞, 1.6)
    """
    
    def __init__(self, region: Region, var_name: str = ""):
        super().__init__()
        self.region = region
        self.var_name = var_name
    
    def semantic_signature(self) -> Tuple:
        # Pure region comparison - var_name is metadata
        return ('region', str(self.region))
    
    def __repr__(self) -> str:
        return f"RegionNode(var={self.var_name}, region={self.region})"


class ActionType(Enum):
    PRINT = "print"
    RETURN = "return"
    ASSIGN = "assign"
    CALL = "call"
    BREAK = "break"
    CONTINUE = "continue"


class ActionNode(CDGNode):
    """
    Represents observable behavior.
    
    Attributes:
        action_type: Type of action (print, return, etc.)
        message: For print - the output message
        value: For return/assign - the value
    """
    
    def __init__(self, action_type: ActionType, message: str = "", value: Any = None):
        super().__init__()
        self.action_type = action_type
        self.message = message
        self.value = value
    
    def semantic_signature(self) -> Tuple:
        if self.action_type == ActionType.PRINT:
            return ('action', 'print', self.message)
        elif self.action_type == ActionType.RETURN:
            return ('action', 'return', str(self.value))
        else:
            return ('action', self.action_type.value)
    
    def __repr__(self) -> str:
        if self.action_type == ActionType.PRINT:
            return f'ActionNode(print, msg="{self.message}")'
        elif self.action_type == ActionType.RETURN:
            return f'ActionNode(return, value={self.value})'
        else:
            return f'ActionNode({self.action_type.value})'


class BodyNode(CDGNode):
    """Root node representing the program body"""
    
    def __init__(self, name: str = "main"):
        super().__init__()
        self.name = name
    
    def semantic_signature(self) -> Tuple:
        child_sigs = tuple(c.semantic_signature() for c in self.children)
        return ('body', child_sigs)
    
    def __repr__(self) -> str:
        return f"BodyNode({self.name})"


# =============================================================================
# CDG BUILDER - Converts Python AST to CDG
# =============================================================================

class CDGBuilder:
    """
    Builds a Cognitive Decision Graph from Python source code.
    """
    
    def __init__(self, code: str):
        self.code = code
        self.tree: Optional[ast.AST] = None
        self.cdg: Optional[BodyNode] = None
        
        # Track variable types from input
        self.var_types: Dict[str, str] = {}
        
        # Track which variable is the "decision variable" for each control structure
        self.current_var: Optional[str] = None
        
        # Track raw input() calls for multi-line input patterns
        # Maps variable name -> prompt string
        self.input_vars: Dict[str, str] = {}
        
        # Track module-level constants for resolving threshold values
        # Maps constant name -> numeric value
        self.constants: Dict[str, float] = {}
        
        self._parse()
    
    def _parse(self):
        """Parse code and build CDG"""
        try:
            self.tree = ast.parse(self.code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {e}")
        
        self._build_cdg()
    
    def _build_cdg(self):
        """Build CDG from AST"""
        self.cdg = BodyNode()
        
        # First pass: scan for module-level constants
        self._scan_constants(self.tree.body)
        
        # Find main function or use module body
        main_body = self.tree.body
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                main_body = node.body
                break
        
        # Also scan for constants inside the function body
        # (handles cases like HEIGHT_MINIMUM = 1.6 inside main())
        self._scan_constants(main_body)
        
        # Process statements
        for stmt in main_body:
            cdg_nodes = self._process_statement(stmt)
            for node in cdg_nodes:
                self.cdg.add_child(node)
        
        # Finalize any unprocessed input() calls (standalone string inputs)
        for input_node in self._finalize_unprocessed_inputs():
            # Insert at the beginning since inputs typically come first conceptually
            self.cdg.children.insert(0, input_node)
            input_node.parent = self.cdg
    
    def _scan_constants(self, body: List[ast.stmt]):
        """
        Scan for module-level constant assignments.
        Constants are identified as UPPERCASE variable names assigned to numeric literals.
        """
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    var_name = stmt.targets[0].id
                    # Check if it looks like a constant (UPPERCASE or contains underscore with uppercase)
                    if var_name.isupper() or (var_name.upper() == var_name and '_' in var_name):
                        # Check if assigned to a numeric literal
                        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, (int, float)):
                            self.constants[var_name] = float(stmt.value.value)
                        # Also handle negative numbers: -1.6
                        elif isinstance(stmt.value, ast.UnaryOp) and isinstance(stmt.value.op, ast.USub):
                            if isinstance(stmt.value.operand, ast.Constant) and isinstance(stmt.value.operand.value, (int, float)):
                                self.constants[var_name] = -float(stmt.value.operand.value)
    
    def _process_statement(self, stmt: ast.stmt) -> List[CDGNode]:
        """Convert an AST statement to CDG node(s)"""
        
        if isinstance(stmt, ast.Assign):
            return self._process_assign(stmt)
        
        elif isinstance(stmt, ast.If):
            return self._process_if(stmt)
        
        elif isinstance(stmt, ast.While):
            return self._process_while(stmt)
        
        elif isinstance(stmt, ast.For):
            return self._process_for(stmt)
        
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call):
                return self._process_call(stmt.value)
        
        elif isinstance(stmt, ast.Return):
            return self._process_return(stmt)
        
        return []
    
    def _process_assign(self, stmt: ast.Assign) -> List[CDGNode]:
        """Process assignment, especially input() calls"""
        nodes = []
        
        if len(stmt.targets) != 1:
            return nodes
        
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return nodes
        
        var_name = target.id
        value = stmt.value
        
        # Check for input patterns: float(input(...)), int(input(...)), input(...)
        input_node = self._extract_input_node(value, var_name)
        if input_node:
            nodes.append(input_node)
        
        return nodes
    
    def _track_raw_input(self, value: ast.expr, var_name: str) -> bool:
        """
        Track raw input() calls for multi-line input patterns.
        Returns True if this is a raw input() call.
        """
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == 'input':
                prompt = ""
                if value.args and isinstance(value.args[0], ast.Constant):
                    prompt = str(value.args[0].value)
                self.input_vars[var_name] = prompt
                return True
        return False
    
    def _check_deferred_type_conversion(self, value: ast.expr, var_name: str) -> Optional[InputNode]:
        """
        Check if this is a type conversion of a previously captured input variable.
        Handles patterns like: height = float(height_str) where height_str = input(...)
        """
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            type_name = value.func.id
            if type_name in ('float', 'int'):
                # Check if the argument is a variable that was assigned from input()
                if value.args and isinstance(value.args[0], ast.Name):
                    input_var = value.args[0].id
                    if input_var in self.input_vars:
                        prompt = self.input_vars[input_var]
                        self.var_types[var_name] = type_name
                        # Remove from input_vars to avoid creating duplicate InputNodes
                        del self.input_vars[input_var]
                        return InputNode(var_name, type_name, prompt)
        return None
    
    def _extract_input_node(self, value: ast.expr, var_name: str) -> Optional[InputNode]:
        """Extract InputNode from assignment value"""
        
        # Pattern 1: float(input("...")) or int(input("...")) - single line
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            type_name = value.func.id
            
            if type_name in ('float', 'int'):
                if value.args and isinstance(value.args[0], ast.Call):
                    inner_call = value.args[0]
                    if isinstance(inner_call.func, ast.Name) and inner_call.func.id == 'input':
                        prompt = ""
                        if inner_call.args and isinstance(inner_call.args[0], ast.Constant):
                            prompt = str(inner_call.args[0].value)
                        
                        self.var_types[var_name] = type_name
                        return InputNode(var_name, type_name, prompt)
                
                # Pattern 2: float(var) or int(var) where var was assigned from input()
                # This handles multi-line input patterns
                deferred_node = self._check_deferred_type_conversion(value, var_name)
                if deferred_node:
                    return deferred_node
            
            # Pattern 3: input("...") directly - track for potential later conversion
            elif type_name == 'input':
                prompt = ""
                if value.args and isinstance(value.args[0], ast.Constant):
                    prompt = str(value.args[0].value)
                
                # Track this for potential multi-line pattern
                # Don't create InputNode yet - wait to see if there's a type conversion
                self._track_raw_input(value, var_name)
                self.var_types[var_name] = 'str'
                # Return None - InputNode will be created when type conversion is found
                # or during finalization for standalone input() calls
                return None
        
        return None
    
    def _finalize_unprocessed_inputs(self) -> List[InputNode]:
        """
        Create InputNodes for input() calls that were never type-converted.
        These are standalone string inputs.
        """
        nodes = []
        for var_name, prompt in self.input_vars.items():
            nodes.append(InputNode(var_name, 'str', prompt))
        self.input_vars.clear()
        return nodes
    
    def _process_if(self, stmt: ast.If) -> List[CDGNode]:
        """Process if/elif/else chain"""
        
        # Detect the decision variable from the first condition
        decision_var = self._extract_decision_variable(stmt.test)
        self.current_var = decision_var
        
        control_node = ControlNode(ControlType.IF, decision_var)
        
        # Track covered regions for computing else branch
        covered_region = Region.empty()
        
        # Process if and elif branches
        current = stmt
        while True:
            # Extract region from condition
            region = self._condition_to_region(current.test)
            
            # Create branch node
            branch = BranchNode(region, is_else=False)
            
            # Process branch body
            for body_stmt in current.body:
                child_nodes = self._process_statement(body_stmt)
                for child in child_nodes:
                    branch.add_child(child)
            
            control_node.add_branch(branch)
            covered_region = covered_region.union(region)
            
            # Check for elif (if inside orelse)
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
            else:
                # Process else branch if present
                if current.orelse:
                    else_region = covered_region.complement()
                    else_branch = BranchNode(else_region, is_else=True)
                    
                    for else_stmt in current.orelse:
                        child_nodes = self._process_statement(else_stmt)
                        for child in child_nodes:
                            else_branch.add_child(child)
                    
                    control_node.add_branch(else_branch)
                break
        
        self.current_var = None
        return [control_node]
    
    def _process_while(self, stmt: ast.While) -> List[CDGNode]:
        """Process while loop"""
        control_node = ControlNode(ControlType.WHILE)
        
        # Check for while True pattern
        if isinstance(stmt.test, ast.Constant) and stmt.test.value == True:
            # Infinite loop - body is the interesting part
            for body_stmt in stmt.body:
                child_nodes = self._process_statement(body_stmt)
                for child in child_nodes:
                    control_node.add_child(child)
        else:
            # Conditional while - extract condition
            decision_var = self._extract_decision_variable(stmt.test)
            control_node.var_name = decision_var
            
            region = self._condition_to_region(stmt.test)
            branch = BranchNode(region)
            
            for body_stmt in stmt.body:
                child_nodes = self._process_statement(body_stmt)
                for child in child_nodes:
                    branch.add_child(child)
            
            control_node.add_branch(branch)
        
        return [control_node]
    
    def _process_for(self, stmt: ast.For) -> List[CDGNode]:
        """Process for loop"""
        control_node = ControlNode(ControlType.FOR)
        
        for body_stmt in stmt.body:
            child_nodes = self._process_statement(body_stmt)
            for child in child_nodes:
                control_node.add_child(child)
        
        return [control_node]
    
    def _process_call(self, call: ast.Call) -> List[CDGNode]:
        """Process function call (especially print)"""
        if isinstance(call.func, ast.Name) and call.func.id == 'print':
            message = ""
            if call.args:
                if isinstance(call.args[0], ast.Constant):
                    message = str(call.args[0].value)
                else:
                    message = "<complex_expr>"
            
            return [ActionNode(ActionType.PRINT, message=message)]
        
        return []
    
    def _process_return(self, stmt: ast.Return) -> List[CDGNode]:
        """Process return statement"""
        value = None
        if stmt.value:
            if isinstance(stmt.value, ast.Constant):
                value = stmt.value.value
            else:
                value = "<complex_expr>"
        
        return [ActionNode(ActionType.RETURN, value=value)]
    
    def _extract_decision_variable(self, node: ast.expr) -> Optional[str]:
        """Extract the variable being tested in a condition"""
        
        if isinstance(node, ast.Compare):
            if isinstance(node.left, ast.Name):
                return node.left.id
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name):
                    return comparator.id
        
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                var = self._extract_decision_variable(value)
                if var:
                    return var
        
        return None
    
    def _condition_to_region(self, node: ast.expr) -> Region:
        """Convert AST condition to Region"""
        
        if isinstance(node, ast.Compare):
            return self._compare_to_region(node)
        
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # AND = intersection of regions
                result = Region.full()
                for value in node.values:
                    sub_region = self._condition_to_region(value)
                    result = result.intersect(sub_region)
                return result
            
            elif isinstance(node.op, ast.Or):
                # OR = union of regions
                result = Region.empty()
                for value in node.values:
                    sub_region = self._condition_to_region(value)
                    result = result.union(sub_region)
                return result
        
        elif isinstance(node, ast.Constant):
            # True/False constants
            if node.value == True:
                return Region.full()
            elif node.value == False:
                return Region.empty()
        
        # Fallback
        return Region.full()
    
    def _compare_to_region(self, node: ast.Compare) -> Region:
        """Convert comparison to Region"""
        
        # Handle chained comparisons: 1.6 < height < 1.9
        result = Region.full()
        
        operands = [node.left] + list(node.comparators)
        
        for left, op, right in zip(operands, node.ops, operands[1:]):
            op_str = self._ast_op_to_str(op)
            
            # Try to get numeric value from either side
            right_value = self._get_numeric_value(right)
            left_value = self._get_numeric_value(left)
            
            # Determine which side has the constant/numeric value
            if right_value is not None:
                # var OP const
                region = Region.from_constraint(op_str, right_value)
            elif left_value is not None:
                # const OP var -> flip
                flipped_op = self._flip_operator(op_str)
                region = Region.from_constraint(flipped_op, left_value)
            else:
                # Non-numeric comparison
                region = Region.full()
            
            result = result.intersect(region)
        
        return result
    
    def _get_numeric_value(self, node: ast.expr) -> Optional[float]:
        """
        Extract numeric value from an AST node.
        Handles:
        - Literal numbers: 1.6, 2
        - Constant variables: HEIGHT_MINIMUM (resolved from self.constants)
        - Negative numbers: -1.6
        """
        # Direct numeric literal
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        
        # Variable reference - check if it's a known constant
        if isinstance(node, ast.Name):
            if node.id in self.constants:
                return self.constants[node.id]
        
        # Negative number: -1.6
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner_value = self._get_numeric_value(node.operand)
            if inner_value is not None:
                return -inner_value
        
        return None
    
    def _ast_op_to_str(self, op: ast.cmpop) -> str:
        """Convert AST comparison operator to string"""
        op_map = {
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Eq: '==',
            ast.NotEq: '!=',
        }
        return op_map.get(type(op), '==')
    
    def _flip_operator(self, op_str: str) -> str:
        """Flip operator when moving variable to left side"""
        flip_map = {
            '<': '>',
            '<=': '>=',
            '>': '<',
            '>=': '<=',
            '==': '==',
            '!=': '!=',
        }
        return flip_map.get(op_str, op_str)
    
    def get_cdg(self) -> BodyNode:
        """Return the built CDG"""
        return self.cdg
    
    def print_cdg(self, node: CDGNode = None, indent: int = 0):
        """Pretty print the CDG tree"""
        if node is None:
            node = self.cdg
        
        prefix = "  " * indent
        print(f"{prefix}{node}")
        
        # Print branches for ControlNode
        if isinstance(node, ControlNode):
            for branch in node.branches:
                self.print_cdg(branch, indent + 1)
        
        # Print children
        for child in node.children:
            self.print_cdg(child, indent + 1)


# =============================================================================
# CDG COMPARISON
# =============================================================================

def cdg_similarity(cdg1: BodyNode, cdg2: BodyNode) -> Dict[str, float]:
    """
    Compare two CDGs and return similarity scores.
    
    Returns:
        Dict with keys:
        - 'structural': How similar the control flow structure is (regions + control flow, NOT actions)
        - 'behavioral': How similar the input/output behavior is
        - 'regional': How similar the decision regions are
        - 'overall': Weighted combination
    """
    
    # Extract specific components for detailed comparison
    inputs1 = _collect_nodes_by_type(cdg1, InputNode)
    inputs2 = _collect_nodes_by_type(cdg2, InputNode)
    
    actions1 = _collect_nodes_by_type(cdg1, ActionNode)
    actions2 = _collect_nodes_by_type(cdg2, ActionNode)
    
    branches1 = _collect_nodes_by_type(cdg1, BranchNode)
    branches2 = _collect_nodes_by_type(cdg2, BranchNode)
    
    controls1 = _collect_nodes_by_type(cdg1, ControlNode)
    controls2 = _collect_nodes_by_type(cdg2, ControlNode)
    
    # Input similarity
    input_sigs1 = {n.semantic_signature() for n in inputs1}
    input_sigs2 = {n.semantic_signature() for n in inputs2}
    input_sim = _jaccard(input_sigs1, input_sigs2)
    
    # Action similarity
    action_sigs1 = {n.semantic_signature() for n in actions1}
    action_sigs2 = {n.semantic_signature() for n in actions2}
    action_sim = _jaccard(action_sigs1, action_sigs2)
    
    # Region similarity (just the regions, ignoring what's inside them)
    region_sigs1 = {str(n.region) for n in branches1}
    region_sigs2 = {str(n.region) for n in branches2}
    region_sim = _jaccard(region_sigs1, region_sigs2)
    
    # Structural similarity: compares control flow structure + regions (NOT action content)
    # This captures whether the decision tree has the same shape
    struct_sig1 = _structural_signature(cdg1)
    struct_sig2 = _structural_signature(cdg2)
    structural_sim = 1.0 if struct_sig1 == struct_sig2 else _structural_similarity(cdg1, cdg2)
    
    # Behavioral = weighted combination of input + action
    behavioral_sim = 0.3 * input_sim + 0.7 * action_sim
    
    # Overall
    overall = 0.2 * structural_sim + 0.3 * behavioral_sim + 0.5 * region_sim
    
    return {
        'structural': structural_sim,
        'behavioral': behavioral_sim,
        'regional': region_sim,
        'input': input_sim,
        'action': action_sim,
        'overall': overall
    }


def _structural_signature(node: CDGNode) -> Any:
    """
    Extract structural signature that captures control flow and regions,
    but NOT action content (print messages, return values, etc.)
    """
    if isinstance(node, BodyNode):
        child_sigs = tuple(sorted(str(_structural_signature(c)) for c in node.children))
        return ('body', child_sigs)
    
    elif isinstance(node, InputNode):
        return ('input', node.input_type)
    
    elif isinstance(node, ControlNode):
        # For control nodes, use frozenset for order-independent branch comparison
        branch_sigs = frozenset(_structural_signature(b) for b in node.branches)
        # Also include children (for while True loops that have content in children, not branches)
        child_sigs = tuple(sorted(str(_structural_signature(c)) for c in node.children))
        return ('control', node.control_type.value, branch_sigs, child_sigs)
    
    elif isinstance(node, BranchNode):
        # For branches, only include region - NOT children (actions)
        return ('branch', str(node.region))
    
    elif isinstance(node, ActionNode):
        # Only include action type, not the message content
        return ('action', node.action_type.value)
    
    else:
        return ('unknown',)


def _structural_similarity(cdg1: BodyNode, cdg2: BodyNode) -> float:
    """
    Compute structural similarity as a continuous score.
    Compares control flow structure and region coverage.
    """
    # Compare control node types
    controls1 = _collect_nodes_by_type(cdg1, ControlNode)
    controls2 = _collect_nodes_by_type(cdg2, ControlNode)
    
    control_types1 = {(c.control_type.value, len(c.branches)) for c in controls1}
    control_types2 = {(c.control_type.value, len(c.branches)) for c in controls2}
    control_sim = _jaccard(control_types1, control_types2)
    
    # Compare branches by region (order-independent)
    branches1 = _collect_nodes_by_type(cdg1, BranchNode)
    branches2 = _collect_nodes_by_type(cdg2, BranchNode)
    
    region_sigs1 = {str(n.region) for n in branches1}
    region_sigs2 = {str(n.region) for n in branches2}
    region_sim = _jaccard(region_sigs1, region_sigs2)
    
    # Compare inputs
    inputs1 = _collect_nodes_by_type(cdg1, InputNode)
    inputs2 = _collect_nodes_by_type(cdg2, InputNode)
    input_types1 = {n.input_type for n in inputs1}
    input_types2 = {n.input_type for n in inputs2}
    input_sim = _jaccard(input_types1, input_types2)
    
    # Weighted combination
    return 0.3 * control_sim + 0.5 * region_sim + 0.2 * input_sim


def _collect_nodes_by_type(root: CDGNode, node_type: type) -> List[CDGNode]:
    """Recursively collect all nodes of a given type"""
    result = []
    
    def visit(node):
        if isinstance(node, node_type):
            result.append(node)
        
        if isinstance(node, ControlNode):
            for branch in node.branches:
                visit(branch)
        
        for child in node.children:
            visit(child)
    
    visit(root)
    return result


def _jaccard(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_cdg(code: str) -> BodyNode:
    """Build a CDG from Python code"""
    builder = CDGBuilder(code)
    return builder.get_cdg()


def compare_programs(code1: str, code2: str) -> Dict[str, float]:
    """Compare two programs and return similarity scores"""
    cdg1 = build_cdg(code1)
    cdg2 = build_cdg(code2)
    return cdg_similarity(cdg1, cdg2)


def print_cdg(code: str):
    """Build and print CDG for a piece of code"""
    builder = CDGBuilder(code)
    builder.print_cdg()

