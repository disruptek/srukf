#!/usr/bin/env python3
"""
Generate SVG performance chart from benchmark output.

Usage: ./benchmark.out | python3 generate_chart.py > benchmark.svg
"""

import sys
import re

def parse_benchmark_output(text):
    """Parse benchmark output into structured data."""
    results = []
    
    # Match lines like: 3x2      predict             0.343        0.065 ...
    # or:                        predict_to          0.337        0.048 ...
    # The dimension may be absent (continued from previous line)
    pattern = r'^\s*(\d+x\d+)?\s*(predict|predict_to|correct|correct_to)\s+([\d.]+)'
    
    current_dim = None
    for line in text.split('\n'):
        match = re.match(pattern, line)
        if match:
            dim, op, mean = match.groups()
            if dim:
                current_dim = dim
            if current_dim:
                results.append({
                    'dim': current_dim,
                    'op': op,
                    'mean_us': float(mean)
                })
    
    return results

def generate_svg(results):
    """Generate SVG bar chart."""
    
    # Filter to just predict and correct (not _to variants)
    data = [r for r in results if r['op'] in ('predict', 'correct')]
    
    # Group by dimension
    dims = []
    seen = set()
    for r in data:
        if r['dim'] not in seen:
            dims.append(r['dim'])
            seen.add(r['dim'])
    
    # Chart dimensions
    margin_left = 60
    margin_right = 20
    margin_top = 60  # Extra room for legend above chart
    margin_bottom = 60
    chart_width = 500
    chart_height = 250
    
    width = margin_left + chart_width + margin_right
    height = margin_top + chart_height + margin_bottom
    
    # Calculate scales
    max_val = max(r['mean_us'] for r in data) * 1.1
    
    bar_group_width = chart_width / len(dims)
    bar_width = bar_group_width * 0.35
    bar_gap = bar_group_width * 0.1
    
    # Colors
    predict_color = "#4a90d9"  # Blue
    correct_color = "#d94a4a"  # Red
    
    # Start SVG
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">')
    
    # Background
    svg.append(f'  <rect width="{width}" height="{height}" fill="#ffffff"/>')
    
    # Title
    svg.append(f'  <text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-weight="600" fill="#333">SR-UKF Performance (lower is better)</text>')
    
    # Grid lines and Y-axis labels
    num_grid = 5
    for i in range(num_grid + 1):
        y = margin_top + chart_height - (i / num_grid) * chart_height
        val = (i / num_grid) * max_val
        
        # Grid line
        svg.append(f'  <line x1="{margin_left}" y1="{y}" x2="{margin_left + chart_width}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>')
        
        # Y-axis label
        svg.append(f'  <text x="{margin_left - 8}" y="{y + 4}" text-anchor="end" font-size="11" fill="#666">{val:.0f}</text>')
    
    # Y-axis title
    svg.append(f'  <text x="15" y="{margin_top + chart_height/2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 15, {margin_top + chart_height/2})">Time (μs)</text>')
    
    # Bars
    for i, dim in enumerate(dims):
        group_x = margin_left + i * bar_group_width + bar_gap
        
        predict_data = next((r for r in data if r['dim'] == dim and r['op'] == 'predict'), None)
        correct_data = next((r for r in data if r['dim'] == dim and r['op'] == 'correct'), None)
        
        if predict_data:
            h = (predict_data['mean_us'] / max_val) * chart_height
            y = margin_top + chart_height - h
            svg.append(f'  <rect x="{group_x}" y="{y}" width="{bar_width}" height="{h}" fill="{predict_color}" rx="2"/>')
            # Value label
            svg.append(f'  <text x="{group_x + bar_width/2}" y="{y - 4}" text-anchor="middle" font-size="9" fill="#666">{predict_data["mean_us"]:.1f}</text>')
        
        if correct_data:
            h = (correct_data['mean_us'] / max_val) * chart_height
            y = margin_top + chart_height - h
            x = group_x + bar_width + bar_gap/2
            svg.append(f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{correct_color}" rx="2"/>')
            # Value label
            svg.append(f'  <text x="{x + bar_width/2}" y="{y - 4}" text-anchor="middle" font-size="9" fill="#666">{correct_data["mean_us"]:.1f}</text>')
        
        # X-axis label (dimension)
        label_x = group_x + bar_width + bar_gap/4
        svg.append(f'  <text x="{label_x}" y="{margin_top + chart_height + 20}" text-anchor="middle" font-size="11" fill="#333">{dim}</text>')
    
    # X-axis title
    svg.append(f'  <text x="{margin_left + chart_width/2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#666">State × Measurement Dimensions</text>')
    
    # Legend (above chart area)
    legend_x = margin_left + chart_width - 130
    legend_y = 38
    svg.append(f'  <rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{predict_color}" rx="2"/>')
    svg.append(f'  <text x="{legend_x + 16}" y="{legend_y + 10}" font-size="11" fill="#333">predict</text>')
    svg.append(f'  <rect x="{legend_x + 70}" y="{legend_y}" width="12" height="12" fill="{correct_color}" rx="2"/>')
    svg.append(f'  <text x="{legend_x + 86}" y="{legend_y + 10}" font-size="11" fill="#333">correct</text>')
    
    # Axes
    svg.append(f'  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#333" stroke-width="1"/>')
    svg.append(f'  <line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#333" stroke-width="1"/>')
    
    svg.append('</svg>')
    
    return '\n'.join(svg)

def main():
    text = sys.stdin.read()
    results = parse_benchmark_output(text)
    
    if not results:
        print("Error: Could not parse benchmark output", file=sys.stderr)
        sys.exit(1)
    
    svg = generate_svg(results)
    print(svg)

if __name__ == '__main__':
    main()
