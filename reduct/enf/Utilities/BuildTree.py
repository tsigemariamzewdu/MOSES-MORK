import re
from typing import Union
from ..DataStructures.Trees import *


def BuildTree(input: str) -> BinaryExpressionTreeNode | None:
    """
    Convert a string of a Boolean expression into a binary expression tree structure.

    Parameters
    ----------
    input : str
        String of Boolean expression.
    
    Returns
    -------
    BinaryExpressionTreeNode | None
        The root node of the binary expression tree.
    
    Raises
    ------
    ValueError
        If the input format is invalid or there are insufficient arguments for a binary operator.
    """
    input = re.sub(r"\s+", "", input)
    if not input:
        return None

    # Drop empty bare operator tokens.
    if input in {"|", "&"}:
        return None

    # Drop  standalone operators that may leak from upstream expression generation.
    if input.upper() in {"OR", "AND"}:
        return None

    tree = BinaryExpressionTreeNode("Null")
    first = input[0]
    
    match first:
        case "|" | "&":
            input = input[2 : len(input) - 1]
            firstArg, secondArg = splitArgs(input)

            if firstArg is None or secondArg is None:
                raise ValueError("Insufficient arguments for binary operator")

            tree.value = "AND" if first == "&" else "OR"
            tree.type = NodeType.AND if first == "&" else NodeType.OR
            tree.left = BuildTree(firstArg)
            tree.right = BuildTree(secondArg)

            # Prune empty children and collapse unary leftovers.
            if tree.left is None and tree.right is None:
                return None
            if tree.left is None:
                return tree.right
            if tree.right is None:
                return tree.left

            return tree
        case "!":
            input = input[2 : len(input) - 1]
            tree.value = "NOT"
            tree.type = NodeType.NOT
            tree.right = BuildTree(input)
            if tree.right is None:
                return None
            return tree
        case "(" | ")":
            raise ValueError("Invalid Boolean expression format")
        case _:
            tree.value = input
            tree.type = NodeType.LITERAL
            return tree


def splitArgs(input: str) -> Union[tuple[str, str], tuple[None, None]]:
    """
    Split a string of a Boolean expression into two arguments.

    Parameters
    ----------
    input : str
        String of Boolean expression.
    
    Returns
    -------
    tuple[str, str] | tuple[None, None]
        A tuple containing the two arguments as strings if split is successful,
        otherwise a tuple of (None, None).
    """
    brackets = 0
    index = 0
    for c in input:
        match c:
            case "(":
                brackets += 1
            case ")":
                brackets -= 1
            case "," if brackets == 0:
                return input[:index], input[index + 1 :]
        index += 1
    return None, None
