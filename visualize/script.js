fetch('graph_data.json')
.then(response => response.json())
.then(graphData => {
    const nodes = new vis.DataSet(graphData.nodes);
    const edges = new vis.DataSet(graphData.edges);

    const container = document.getElementById('graph');
    
                // Color nodes and edges based on types
                nodes.forEach(node => {
        if (node.shape === 'ellipse') {
        node.color = '#FF6347';  // Tomato color for MessageNodes
        } else if (node.shape === 'box') {
        node.color = '#4682B4';  // SteelBlue color for KeywordNodes
        }
    });

    edges.forEach(edge => {
        if (edge.label === 'describes') {
        edge.color = { color: '#32CD32' };  // Green color for "describes"
        edge.width = 2;
        } else if (edge.label === 'follows') {
        edge.color = { color: '#FFD700' };  // Yellow for "follows"
        edge.width = 2;
        } else if (edge.label === 'refers') {
        edge.color = { color: '#8A2BE2' };  // BlueViolet for "refers"
        edge.width = 2;
        }
    });

    // You can adjust more options to fine-tune the graph

    const data = {
        nodes: nodes,
        edges: edges
    };

    const options = {
        nodes: {
        shape: 'dot',
        size: 15,
        font: {
            size: 12,
            color: '#ffffff'
        }
        },
        edges: {
        width: 2,
        font: {
            size: 10,
            align: 'middle'
        },
        // smooth: {
        //   type: 'continuous'
        // }
        smooth: {
            enabled: false  // Disable curved edges
        },
        arrows: {
            to: {  // Add an arrowhead to the target node
                enabled: true,
                scaleFactor: 1
            }
        }
        },
        interaction: {
        hover: true
        },
        physics: {
        enabled: false,
        barnesHut: {
            gravitationalConstant: -8000,
            centralGravity: 0.1,
            springLength: 200,
            springConstant: 0.04
        }
        }
    };

    // Create network
    const network = new vis.Network(container, data, options);


    // Search functionality
    document.getElementById('search-button').addEventListener('click', () => {
      const query = document.getElementById('search-bar').value.toLowerCase();
      if (!query) return;

      // Find matching nodes
      const matchingNodes = nodes.get().filter(node =>
        node.label.toLowerCase().includes(query) || node.title.toLowerCase().includes(query)
      );

      if (matchingNodes.length === 0) {
        alert('No matching nodes found!');
        return;
      }

      // Highlight and focus on the first matching node
      const firstMatch = matchingNodes[0];
      nodes.update({ id: firstMatch.id, color: { background: 'yellow', border: 'orange' }, size: 20 });

      network.focus(firstMatch.id, {
        scale: 1.5,  // Zoom in on the node
        animation: { duration: 1000 }
      });

      // Optional: Reset other nodes
      setTimeout(() => {
        nodes.update({ id: firstMatch.id, color: null, size: 15 });
      }, 3000);  // Reset after 3 seconds
    });
  })
  .catch(error => console.error('Error loading the graph data:', error));
