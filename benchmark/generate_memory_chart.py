#!/usr/bin/env python3
"""
Generate SVG memory chart from memory benchmark output.

Usage: ./memory_bench.out | python3 generate_memory_chart.py > memory.svg
"""

import sys
import re

def parse_benchmark_output(text):
    """Parse memory benchmark output into structured data."""
    results = []
    
    # Match lines like: 3x2                0.58         3.42         4.00         8.00
    pattern = r'^\s*(\d+x\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    
    for line in text.split('\n'):
        match = re.match(pattern, line)
        if match:
            dim, filter_kb, workspace_kb, total_calc, total_meas = match.groups()
            results.append({
                'dim': dim,
                'filter_kb': float(filter_kb),
                'workspace_kb': float(workspace_kb),
                'total_calc_kb': float(total_calc),
                'total_meas_kb': float(total_meas)
            })
    
    return results

def generate_svg(results):
    """Generate SVG chart with both calculated and measured memory."""
    
    # Use only the first 5 dimensions (same as CPU benchmark for consistency)
    data = results[:5]
    
    # Chart dimensions
    margin_left = 70
    margin_right = 20
    margin_top = 60
    margin_bottom = 60
    chart_width = 500
    chart_height = 250
    
    width = margin_left + chart_width + margin_right
    height = margin_top + chart_height + margin_bottom
    
    # Calculate scales
    max_val = max(max(r['total_calc_kb'], r['total_meas_kb']) for r in data) * 1.15
    
    bar_group_width = chart_width / len(data)
    bar_width = bar_group_width * 0.35
    bar_gap = bar_group_width * 0.1
    
    # Colors - calculated is primary (solid), measured is secondary (lighter)
    calc_color = "#2d7d2d"    # Dark green for calculated (authoritative)
    meas_color = "#b0b0b0"    # Gray for measured (noisy/lazy alloc)
    
    # Start SVG
    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, sans-serif">')
    
    # Background
    svg.append(f'  <rect width="{width}" height="{height}" fill="#ffffff"/>')
    
    # Title
    svg.append(f'  <text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-weight="600" fill="#333">SR-UKF Memory Usage</text>')
    
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
    svg.append(f'  <text x="18" y="{margin_top + chart_height/2}" text-anchor="middle" font-size="12" fill="#666" transform="rotate(-90, 18, {margin_top + chart_height/2})">Memory (KB)</text>')
    
    # Bars
    for i, r in enumerate(data):
        group_x = margin_left + i * bar_group_width + bar_gap
        
        # Calculated (green)
        h = (r['total_calc_kb'] / max_val) * chart_height
        y = margin_top + chart_height - h
        svg.append(f'  <rect x="{group_x}" y="{y}" width="{bar_width}" height="{h}" fill="{calc_color}" rx="2"/>')
        svg.append(f'  <text x="{group_x + bar_width/2}" y="{y - 4}" text-anchor="middle" font-size="9" fill="#666">{r["total_calc_kb"]:.0f}</text>')
        
        # Measured (purple) - only if non-zero
        if r['total_meas_kb'] > 0:
            h = (r['total_meas_kb'] / max_val) * chart_height
            y = margin_top + chart_height - h
            x = group_x + bar_width + bar_gap/2
            svg.append(f'  <rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{meas_color}" rx="2"/>')
            svg.append(f'  <text x="{x + bar_width/2}" y="{y - 4}" text-anchor="middle" font-size="9" fill="#666">{r["total_meas_kb"]:.0f}</text>')
        
        # X-axis label
        label_x = group_x + bar_width + bar_gap/4
        svg.append(f'  <text x="{label_x}" y="{margin_top + chart_height + 20}" text-anchor="middle" font-size="11" fill="#333">{r["dim"]}</text>')
    
    # X-axis title
    svg.append(f'  <text x="{margin_left + chart_width/2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#666">State × Measurement Dimensions</text>')
    
    # Legend
    legend_x = margin_left + chart_width - 145
    legend_y = 38
    svg.append(f'  <rect x="{legend_x}" y="{legend_y}" width="12" height="12" fill="{calc_color}" rx="2"/>')
    svg.append(f'  <text x="{legend_x + 16}" y="{legend_y + 10}" font-size="11" fill="#333">allocated</text>')
    svg.append(f'  <rect x="{legend_x + 75}" y="{legend_y}" width="12" height="12" fill="{meas_color}" rx="2"/>')
    svg.append(f'  <text x="{legend_x + 91}" y="{legend_y + 10}" font-size="11" fill="#333">resident</text>')
    
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
