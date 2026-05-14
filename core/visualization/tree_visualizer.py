"""
多源聚合树可视化模块
支持生成 HTML 交互式图表
"""
from typing import Dict, List
from core.history.tree_history import MultiSourceTree, QANode
import json
import os


class TreeVisualizer:
    """多源聚合树可视化器"""
    
    def __init__(self, tree: MultiSourceTree):
        self.tree = tree
    
    def generate_html(self, output_path: str = "tree_visualization.html",
                     max_nodes: int = 50) -> str:
        """
        生成可视化 HTML 文件
        
        Args:
            output_path: 输出文件路径
            max_nodes: 最大显示节点数（避免过载）
        
        Returns:
            HTML 文件路径
        """
        # 生成节点和边的数据
        nodes_data, edges_data = self._build_graph_data(max_nodes)
        
        # 生成统计信息
        stats = self._generate_stats()
        
        # 生成 HTML
        html_content = self._generate_html(nodes_data, edges_data, stats)
        
        # 保存文件
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[可视化] 已生成: {output_path}")
        return output_path
    
    def _build_graph_data(self, max_nodes: int = 50):
        """
        构建图形数据（节点和边）
        
        Returns:
            (nodes, edges) 元组
        """
        nodes = []
        edges = []
        node_count = 0
        
        # 添加信息节点
        for info in self.tree.info_nodes:
            if node_count >= max_nodes:
                break
            
            # 截断内容用于显示
            content_preview = info.content[:80] + "..." if len(info.content) > 80 else info.content
            
            nodes.append({
                'id': info.node_id,
                'label': f"[信息] {content_preview}",
                'type': 'info',
                'content': info.content,
                'source_id': info.source_id,
                'mention_count': info.mention_count,
                'importance': info.importance,
                'merged_count': len(info.merged_from),
                'color': '#4CAF50',  # 绿色
                'shape': 'box'
            })
            node_count += 1
        
        # 添加问答节点
        for qa in self.tree.qa_nodes:
            if node_count >= max_nodes:
                break
            
            question_preview = qa.question[:60] + "..." if len(qa.question) > 60 else qa.question
            
            nodes.append({
                'id': qa.node_id,
                'label': f"[问答] {question_preview}",
                'type': 'qa',
                'question': qa.question,
                'answer': qa.answer[:200],
                'timestamp': qa.timestamp,
                'color': '#2196F3',  # 蓝色
                'shape': 'ellipse'
            })
            node_count += 1
        
        # 添加边（关系）
        for info in self.tree.info_nodes:
            for child_qa in info.children:
                edges.append({
                    'from': info.node_id,
                    'to': child_qa.node_id,
                    'label': '支撑',
                    'color': '#666666'
                })
        
        for qa in self.tree.qa_nodes:
            for parent in qa.parents:
                if isinstance(parent, QANode):
                    edges.append({
                        'from': parent.node_id,
                        'to': qa.node_id,
                        'label': '追问',
                        'color': '#999999'
                    })
        
        return nodes, edges
    
    def _generate_stats(self) -> Dict:
        """生成统计信息"""
        total_info = len(self.tree.info_nodes)
        total_qa = len(self.tree.qa_nodes)
        total_merged = sum(1 for info in self.tree.info_nodes if info.merged_from)
        avg_mention = (sum(info.mention_count for info in self.tree.info_nodes) / total_info 
                      if total_info > 0 else 0)
        
        # 计算连接密度
        total_edges = sum(len(info.children) for info in self.tree.info_nodes)
        total_edges += sum(len(qa.parents) for qa in self.tree.qa_nodes)
        
        return {
            'info_nodes': total_info,
            'qa_nodes': total_qa,
            'merged_nodes': total_merged,
            'avg_mentions': round(avg_mention, 2),
            'total_edges': total_edges,
            'tree_depth': self._calculate_depth()
        }
    
    def _calculate_depth(self) -> int:
        """计算树的最大深度"""
        if not self.tree.qa_nodes:
            return 0
        
        max_depth = 0
        for qa in self.tree.qa_nodes:
            depth = self._get_node_depth(qa, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _get_node_depth(self, node: QANode, visited: set) -> int:
        """递归计算节点深度"""
        if node.node_id in visited:
            return 0
        visited.add(node.node_id)
        
        if not node.parents or all(isinstance(p, type(None)) for p in node.parents):
            return 1
        
        max_parent_depth = 0
        for parent in node.parents:
            if isinstance(parent, QANode):
                parent_depth = self._get_node_depth(parent, visited.copy())
                max_parent_depth = max(max_parent_depth, parent_depth)
        
        return max_parent_depth + 1
    
    def _generate_html(self, nodes: List[Dict], edges: List[Dict], stats: Dict) -> str:
        """生成完整的 HTML 内容"""
        
        # 转换为 JSON
        nodes_json = json.dumps(nodes, ensure_ascii=False)
        edges_json = json.dumps(edges, ensure_ascii=False)
        stats_json = json.dumps(stats, ensure_ascii=False)
        
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多源聚合树可视化</title>
    <script src="https://unpkg.com/vis-network@9.1.2/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 32px;
            font-weight: 600;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            padding: 20px 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        
        #network {{
            width: 100%;
            height: 700px;
            border-top: 2px solid #eee;
        }}
        
        .legend {{
            padding: 20px 30px;
            background: #f8f9fa;
            display: flex;
            gap: 30px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .legend-color {{
            width: 30px;
            height: 30px;
            border-radius: 5px;
        }}
        
        .info {{
            padding: 20px 30px;
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            margin: 20px 30px;
            border-radius: 5px;
        }}
        
        .info strong {{
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌳 多源聚合树可视化</h1>
            <p>信息节点与问答关系图谱</p>
        </div>
        
        <div class="stats" id="stats"></div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4CAF50;"></div>
                <span>信息节点（文档片段）</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #2196F3; border-radius: 50%;"></div>
                <span>问答节点</span>
            </div>
            <div class="legend-item">
                <div style="width: 30px; height: 3px; background: #666;"></div>
                <span>支撑关系</span>
            </div>
            <div class="legend-item">
                <div style="width: 30px; height: 3px; background: #999; border-top: 2px dashed #999;"></div>
                <span>追问关系</span>
            </div>
        </div>
        
        <div class="info">
            <strong>💡 使用说明：</strong>
            鼠标滚轮缩放 · 拖拽移动 · 点击节点查看详情 · 悬停显示关系
        </div>
        
        <div id="network"></div>
    </div>
    
    <script>
        // 统计数据
        const stats = {stats_json};
        
        // 渲染统计卡片
        const statsContainer = document.getElementById('stats');
        const statsData = [
            {{value: stats.info_nodes, label: '信息节点'}},
            {{value: stats.qa_nodes, label: '问答节点'}},
            {{value: stats.merged_nodes, label: '融合节点'}},
            {{value: stats.avg_mentions, label: '平均引用'}},
            {{value: stats.total_edges, label: '关系边'}},
            {{value: stats.tree_depth, label: '树深度'}}
        ];
        
        statsData.forEach(stat => {{
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = `
                <div class="stat-value">${{stat.value}}</div>
                <div class="stat-label">${{stat.label}}</div>
            `;
            statsContainer.appendChild(card);
        }});
        
        // 节点数据
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        
        // 创建 vis.js 数据集
        const nodes = new vis.DataSet(
            nodesData.map(n => ({{
                id: n.id,
                label: n.label,
                color: {{
                    background: n.color,
                    border: n.color,
                    highlight: {{
                        background: '#ff9800',
                        border: '#ff9800'
                    }}
                }},
                shape: n.shape,
                font: {{
                    size: 12,
                    face: 'Segoe UI'
                }},
                borderWidth: 2,
                shadow: true,
                title: generateTooltip(n)
            }}))
        );
        
        const edges = new vis.DataSet(
            edgesData.map(e => ({{
                from: e.from,
                to: e.to,
                label: e.label,
                color: {{
                    color: e.color,
                    highlight: '#ff9800'
                }},
                font: {{
                    size: 10,
                    align: 'top'
                }},
                arrows: 'to',
                smooth: {{
                    type: 'curvedCW',
                    roundness: 0.2
                }}
            }}))
        );
        
        // 生成工具提示
        function generateTooltip(node) {{
            if (node.type === 'info') {{
                return `
                    <div style="padding: 10px; max-width: 300px;">
                        <strong>信息节点</strong><br/>
                        <br/>
                        <strong>内容:</strong> ${{node.content}}<br/>
                        <br/>
                        <strong>来源:</strong> ${{node.source_id}}<br/>
                        <strong>引用次数:</strong> ${{node.mention_count}}<br/>
                        <strong>重要性:</strong> ${{node.importance.toFixed(2)}}<br/>
                        <strong>融合源数:</strong> ${{node.merged_count}}
                    </div>
                `;
            }} else {{
                return `
                    <div style="padding: 10px; max-width: 300px;">
                        <strong>问答节点</strong><br/>
                        <br/>
                        <strong>问题:</strong> ${{node.question}}<br/>
                        <br/>
                        <strong>答案:</strong> ${{node.answer}}<br/>
                        <br/>
                        <strong>时间戳:</strong> ${{node.timestamp}}
                    </div>
                `;
            }}
        }}
        
        // 配置选项
        const options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.05
                }},
                stabilization: {{
                    iterations: 200
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                zoomView: true,
                dragView: true,
                dragNodes: true
            }}
        }};
        
        // 创建网络图
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const network = new vis.Network(container, data, options);
        
        // 点击事件
        network.on("click", function (params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Selected node:', node);
            }}
        }});
    </script>
</body>
</html>"""
        
        return html


def visualize_tree(tree: MultiSourceTree, output_path: str = "tree_visualization.html",
                   max_nodes: int = 50) -> str:
    """
    便捷函数：可视化多源聚合树
    
    Args:
        tree: MultiSourceTree 实例
        output_path: 输出 HTML 文件路径
        max_nodes: 最大显示节点数
    
    Returns:
        输出文件路径
    """
    visualizer = TreeVisualizer(tree)
    return visualizer.generate_html(output_path, max_nodes)
