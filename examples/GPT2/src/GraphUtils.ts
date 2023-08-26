import { Tensor, Matrix } from "micrograd-ts";

export class GraphUtils {
    static PrintMatrix(m: Matrix) {
        return `Matrix(shape=[${m.shape}], mean=${Matrix.mean(m).data[0].toFixed(4)})`;
    }

    static trace(root) {
        const nodes = new Set<Matrix>();
        const edges = new Set<Matrix>();

        function build(v) {
            if (!nodes.has(v)) {
                nodes.add(v);
                for (let child of v._prev) {
                    edges.add([child, v]);
                    build(child);
                }
            }
        }

        build(root);
        return [nodes, edges];
    }

    static draw_dot(root) {
        let str = `digraph g {rankdir="TB"; nodesep=0.5;\n`;
        const [nodes, edges] = GraphUtils.trace(root);

        // Edge creation for nodes to ops
        for (let node of nodes) {
            const nodeName = node.id;
            str += `"${nodeName}" [label="{data (${GraphUtils.PrintMatrix(node.data)}) | grad (${GraphUtils.PrintMatrix(node.grad)})}"] [shape=record, width=5];\n`;
            if (node._op !== "") {
                str += `"${nodeName + node._op}" [label="${node._op}"]\n`;
                str += `"${nodeName + node._op}" -> "${nodeName}";\n`;
            }
        }

        for (let edge of edges) {
            const n1 = edge[0];
            const n2 = edge[1];

            str += `"${n1.id}" -> "${n2.id + n2._op}"\n`
        }

        str += "}"

        // console.log(str);

        var image = Viz(str, { format: 'svg' });
        const imgElement = document.createElement("div");
        imgElement.innerHTML = image;

        return imgElement;
    }
}