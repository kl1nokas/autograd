from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        name = getattr(n, "name", "")
        label = f"{name}\ndata={n.data}\ngrad={n.grad}\nshape={n.data.shape}"

        dot.node(
            name=uid,
            label=label,
            shape='box',
            style='filled',
            fillcolor='lightblue',
            tooltip=label,
            URL=f"javascript:alert('{label}')"
        )

        if n.op:
            op_id = uid + n.op
            dot.node(op_id, label=n.op, shape='circle', color='orange')
            dot.edge(op_id, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot

from engine import Tensor

x = Tensor([[2.0]])
x.name = "x"

y = Tensor([[3.0]])
y.name = "y"

w = Tensor([[4.0]])
w.name = "w"

z = (x + y) * w
z.name = "z"

out = z.relu()
out.name = "out"

out.backward()

dot = draw_dot(out)

draw_dot(out)
