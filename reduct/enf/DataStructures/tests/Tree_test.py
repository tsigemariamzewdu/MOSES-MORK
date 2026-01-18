import unittest

from reduct.enf.DataStructures.Trees import (
    NodeType,
    BinaryExpressionTreeNode,
    TreeNode,
    findAndRemoveChild,
)


class TestNodeType(unittest.TestCase):
    def test_enum_members(self):
        self.assertEqual(NodeType.AND.value, "AND")
        self.assertEqual(NodeType.OR.value, "OR")
        self.assertEqual(NodeType.NOT.value, "NOT")
        self.assertEqual(NodeType.LITERAL.value, "LITERAL")
        self.assertEqual(NodeType.ROOT.value, "ROOT")


class TestBinaryExpressionTreeNode(unittest.TestCase):
    def test_initialization_defaults(self):
        node = BinaryExpressionTreeNode("x")
        self.assertEqual(node.value, "x")
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)
        self.assertEqual(node.type, NodeType.LITERAL)

    def test_children_assignment(self):
        root = BinaryExpressionTreeNode("root")
        left = BinaryExpressionTreeNode("left")
        right = BinaryExpressionTreeNode("right")

        root.left = left
        root.right = right

        self.assertIs(root.left, left)
        self.assertIs(root.right, right)


class TestTreeNode(unittest.TestCase):
    def test_initialization_defaults(self):
        node = TreeNode("A")
        self.assertEqual(node.value, "A")
        self.assertFalse(node.constraint)
        self.assertIsNone(node.left)
        self.assertIsNone(node.right)
        self.assertEqual(node.guardSet, [])
        self.assertEqual(node.children, [])
        self.assertEqual(node.type, NodeType.LITERAL)

    def test_initialization_with_constraint(self):
        node = TreeNode("B", constraint=True)
        self.assertEqual(node.value, "B")
        self.assertTrue(node.constraint)

    def test_hash_and_equality_same(self):
        n1 = TreeNode("A", constraint=True)
        n2 = TreeNode("A", constraint=True)
        n3 = TreeNode("A", constraint=False)

        self.assertEqual(n1, n2)
        self.assertEqual(hash(n1), hash(n2))
        self.assertNotEqual(n1, n3)

    def test_equality_different_type(self):
        n1 = TreeNode("A", constraint=True)
        other = "not_a_tree_node"
        self.assertNotEqual(n1, other)

    def test_repr_contains_fields(self):
        node = TreeNode("A", constraint=True)
        r = repr(node)
        self.assertIn("A", r)
        self.assertIn("constraint=True", r)
        self.assertIn("type=", r)

    def test_node_children_and_guardset(self):
        parent = TreeNode("AND")
        child1 = TreeNode("A")
        child2 = TreeNode("B")
        guard = TreeNode("AND", constraint=True)

        parent.children.append(child1)
        parent.children.append(child2)
        parent.guardSet.append(guard)

        self.assertIn(child1, parent.children)
        self.assertIn(child2, parent.children)
        self.assertIn(guard, parent.guardSet)

    def test_left_right_assignment(self):
        root = TreeNode("root")
        left = TreeNode("left")
        right = TreeNode("right")

        root.left = left
        root.right = right

        self.assertIs(root.left, left)
        self.assertIs(root.right, right)


class TestFindAndRemoveChild(unittest.TestCase):
    def test_remove_from_empty_list(self):
        node = TreeNode("A")
        result = findAndRemoveChild([], node)
        self.assertEqual(result, [])

    def test_remove_existing_child_by_identity(self):
        child1 = TreeNode("A")
        child2 = TreeNode("B")
        target = TreeNode("C")
        children = [child1, target, child2]

        result = findAndRemoveChild(children, target)

        self.assertEqual(len(result), 2)
        self.assertNotIn(target, result)
        self.assertIn(child1, result)
        self.assertIn(child2, result)

    def test_remove_only_first_occurrence_by_identity(self):
        # two distinct objects with same value/constraint
        target1 = TreeNode("A")
        target2 = TreeNode("A")
        other = TreeNode("B")
        children = [target1, target2, other]

        result = findAndRemoveChild(children, target1)

        # first dup removed, second dup kept
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], target2)
        self.assertIs(result[1], other)

    def test_no_removal_when_no_identity_match(self):
        target = TreeNode("A")
        # same value, but not the same object
        same_value = TreeNode("A")
        children = [same_value]

        result = findAndRemoveChild(children, target)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], same_value)


if __name__ == "__main__":
    unittest.main()