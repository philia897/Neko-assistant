import sqlite3
import json
import os
import http.server
import socketserver

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

def start_server():
    # Change the working directory to the script's directory
    os.chdir(script_dir)

    # Set the port you want to use for the server
    PORT = 8000

    # Define the handler for the server (it will serve files from the current directory)
    Handler = http.server.SimpleHTTPRequestHandler

    # Set up the HTTP server
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}/memory_graph.html")
        httpd.serve_forever()

def export_graph_to_json(db_path: str, output_json_path: str):
    """
    Exports the graph data from SQLite to a JSON file for visualization with vis.js.
    :param db_path: Path to the SQLite database file.
    :param output_json_path: Path to save the output JSON file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query keywords
    cursor.execute("SELECT keyword_id, keyword, content FROM keywords")
    nodes = []
    for row in cursor.fetchall():
        nodes.append({
            "id": row[0],
            "label": f"{row[1]}",
            "title": f"Content: {row[2]}",  # Tooltip with content
            "shape": "ellipse",  # Shape for keyword nodes
        })

    # Query message blocks
    cursor.execute("SELECT block_id, messages, title, timestamp FROM message_blocks")
    for row in cursor.fetchall():
        l = min(len(row[1]), 60)
        nodes.append({
            "id": row[0],
            "label": f"{row[2] or row[1][:l]}",
            "title": f"Timestamp: {row[3]}\n{row[1]}",  # Tooltip with messages and timestamp
            "shape": "box",  # Shape for message blocks
        })

    # Query relations
    cursor.execute("SELECT relation_id, source, target, relation_type, relation_desc FROM relations")
    edges = []
    for row in cursor.fetchall():
        edges.append({
            "id": row[0],
            "from": row[1],
            "to": row[2],
            "label": row[3],  # Relation type as label
            "title": f"{row[4]}",  # Tooltip with relation details
        })

    # Create JSON structure for graph
    graph_data = {"nodes": nodes, "edges": edges}

    # Save graph data as JSON
    with open(output_json_path, "w") as f:
        json.dump(graph_data, f, indent=4)

    conn.close()
    print(f"Graph data exported to {output_json_path}")

if __name__ == "__main__":
    export_graph_to_json("memory.db", "visualize/graph_data.json")
    start_server()