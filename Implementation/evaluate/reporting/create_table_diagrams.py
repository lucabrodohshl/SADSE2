"""
Create a visual comparison figure showing Table 1 vs Table 2 structure.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from paths import FIGURES_DIR

def create_table_comparison_diagram():
    """Create a clear diagram explaining the two-table structure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Table 1 - Global/Strategy-Independent
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Title
    ax1.text(5, 9.5, 'Table 1: Cache System Characteristics', 
             ha='center', va='top', fontsize=16, fontweight='bold', color='darkblue')
    ax1.text(5, 9, '(GLOBAL - Strategy-Independent)', 
             ha='center', va='top', fontsize=12, style='italic', color='blue')
    
    # Main box
    rect1 = mpatches.FancyBboxPatch((0.5, 1), 9, 7, 
                                     boxstyle="round,pad=0.1", 
                                     edgecolor='darkblue', facecolor='lightblue', 
                                     linewidth=2, alpha=0.3)
    ax1.add_patch(rect1)
    
    # Content
    y_pos = 7.5
    ax1.text(5, y_pos, '📊 What it shows:', ha='center', fontsize=13, fontweight='bold')
    
    metrics = [
        '• Avg Cache Entries (~11-12)',
        '• Avg Cache Size (~10-13 KB)',
        '• Avg Entry Size (~0.86-1.13 KB)',
        '• Cache Overhead (<20 KB)',
    ]
    
    y_pos -= 0.8
    for metric in metrics:
        ax1.text(5, y_pos, metric, ha='center', fontsize=11)
        y_pos -= 0.6
    
    y_pos -= 0.4
    ax1.text(5, y_pos, '✅ Key Message:', ha='center', fontsize=12, fontweight='bold', color='green')
    y_pos -= 0.6
    ax1.text(5, y_pos, '"Cache overhead is NEGLIGIBLE"', ha='center', fontsize=11, 
             style='italic', color='darkgreen')
    y_pos -= 0.5
    ax1.text(5, y_pos, '"Does NOT vary with scenarios"', ha='center', fontsize=11,
             style='italic', color='darkgreen')
    
    y_pos -= 0.8
    ax1.text(5, y_pos, '🎯 Purpose:', ha='center', fontsize=12, fontweight='bold', color='purple')
    y_pos -= 0.5
    ax1.text(5, y_pos, 'Show cache is lightweight & scalable', ha='center', fontsize=10, 
             color='purple')
    
    # Table 2 - Scenario-Dependent
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Title
    ax2.text(5, 9.5, 'Table 2: Performance Metrics', 
             ha='center', va='top', fontsize=16, fontweight='bold', color='darkred')
    ax2.text(5, 9, '(SCENARIO-DEPENDENT)', 
             ha='center', va='top', fontsize=12, style='italic', color='red')
    
    # Main box
    rect2 = mpatches.FancyBboxPatch((0.5, 1), 9, 7, 
                                     boxstyle="round,pad=0.1", 
                                     edgecolor='darkred', facecolor='lightcoral', 
                                     linewidth=2, alpha=0.3)
    ax2.add_patch(rect2)
    
    # Content
    y_pos = 7.5
    ax2.text(5, y_pos, '📊 What it shows:', ha='center', fontsize=13, fontweight='bold')
    
    metrics2 = [
        '• Cache Hit Rate (68% - 98%)',
        '• Speedup Factor (3× - 59×)',
        '• Latency P50/P95/P99',
        '• Coverage Achieved',
    ]
    
    y_pos -= 0.8
    for metric in metrics2:
        ax2.text(5, y_pos, metric, ha='center', fontsize=11)
        y_pos -= 0.6
    
    y_pos -= 0.4
    ax2.text(5, y_pos, '✅ Key Message:', ha='center', fontsize=12, fontweight='bold', color='green')
    y_pos -= 0.6
    ax2.text(5, y_pos, '"V1: High hit rate (95%) + coarse"', ha='center', fontsize=11, 
             style='italic', color='darkgreen')
    y_pos -= 0.5
    ax2.text(5, y_pos, '"V2: Good hit rate (81%) + precise"', ha='center', fontsize=11,
             style='italic', color='darkgreen')
    
    y_pos -= 0.8
    ax2.text(5, y_pos, '🎯 Purpose:', ha='center', fontsize=12, fontweight='bold', color='purple')
    y_pos -= 0.5
    ax2.text(5, y_pos, 'Show trade-offs & scenario impact', ha='center', fontsize=10, 
             color='purple')
    
    plt.tight_layout()
    return fig


def create_data_flow_diagram():
    """Create a diagram showing what data goes where."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Data Organization: Global vs Scenario-Dependent', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Raw Data Source
    data_box = mpatches.FancyBboxPatch((5, 8), 4, 0.8,
                                        boxstyle="round,pad=0.05",
                                        edgecolor='black', facecolor='lightgray',
                                        linewidth=2)
    ax.add_patch(data_box)
    ax.text(7, 8.4, 'Raw Experiment Results', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow down
    ax.arrow(7, 8, 0, -0.5, head_width=0.3, head_length=0.1, fc='black', ec='black')
    
    # Split
    ax.text(7, 7.2, 'Separate into:', ha='center', fontsize=11, style='italic')
    
    # Left branch - Global
    ax.arrow(7, 7, -2, -0.8, head_width=0.2, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    
    global_box = mpatches.FancyBboxPatch((1, 4), 4, 2,
                                          boxstyle="round,pad=0.1",
                                          edgecolor='darkblue', facecolor='lightblue',
                                          linewidth=2, alpha=0.5)
    ax.add_patch(global_box)
    
    ax.text(3, 5.7, 'GLOBAL METRICS', ha='center', fontsize=13, fontweight='bold', color='darkblue')
    ax.text(3, 5.3, '(Strategy-Independent)', ha='center', fontsize=10, style='italic', color='blue')
    
    global_items = [
        'Cache entries',
        'Cache size (KB)',
        'Entry size (KB)',
        'Memory overhead'
    ]
    
    y_pos = 4.9
    for item in global_items:
        ax.text(3, y_pos, f'• {item}', ha='center', fontsize=9)
        y_pos -= 0.3
    
    ax.arrow(3, 4, 0, -0.5, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    
    table1_box = mpatches.FancyBboxPatch((1.5, 2), 3, 1.2,
                                          boxstyle="round,pad=0.05",
                                          edgecolor='darkblue', facecolor='lightblue',
                                          linewidth=3)
    ax.add_patch(table1_box)
    ax.text(3, 2.8, 'TABLE 1', ha='center', fontsize=14, fontweight='bold', color='darkblue')
    ax.text(3, 2.4, 'Cache Characteristics', ha='center', fontsize=10, color='blue')
    
    # Right branch - Scenario-Dependent
    ax.arrow(7, 7, 2, -0.8, head_width=0.2, head_length=0.1, fc='red', ec='red', linewidth=2)
    
    scenario_box = mpatches.FancyBboxPatch((9, 4), 4, 2,
                                            boxstyle="round,pad=0.1",
                                            edgecolor='darkred', facecolor='lightcoral',
                                            linewidth=2, alpha=0.5)
    ax.add_patch(scenario_box)
    
    ax.text(11, 5.7, 'SCENARIO METRICS', ha='center', fontsize=13, fontweight='bold', color='darkred')
    ax.text(11, 5.3, '(Varies by Conditions)', ha='center', fontsize=10, style='italic', color='red')
    
    scenario_items = [
        'Hit rate (%)',
        'Speedup factor (×)',
        'Latency percentiles',
        'Coverage achieved'
    ]
    
    y_pos = 4.9
    for item in scenario_items:
        ax.text(11, y_pos, f'• {item}', ha='center', fontsize=9)
        y_pos -= 0.3
    
    ax.arrow(11, 4, 0, -0.5, head_width=0.2, head_length=0.1, fc='red', ec='red')
    
    table2_box = mpatches.FancyBboxPatch((9.5, 2), 3, 1.2,
                                          boxstyle="round,pad=0.05",
                                          edgecolor='darkred', facecolor='lightcoral',
                                          linewidth=3)
    ax.add_patch(table2_box)
    ax.text(11, 2.8, 'TABLE 2', ha='center', fontsize=14, fontweight='bold', color='darkred')
    ax.text(11, 2.4, 'Performance Metrics', ha='center', fontsize=10, color='red')
    
    # Bottom legend
    legend_y = 1
    ax.text(7, legend_y, '💡 Key Insight:', ha='center', fontsize=12, fontweight='bold')
    ax.text(7, legend_y - 0.4, 'Table 1 shows cache is lightweight (global property)', 
            ha='center', fontsize=10, style='italic')
    ax.text(7, legend_y - 0.7, 'Table 2 shows cache effectiveness varies with scenarios (local property)', 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig


def main():
    output_dir = FIGURES_DIR
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("GENERATING TABLE COMPARISON DIAGRAMS")
    print("="*80 + "\n")
    
    # Diagram 1: Table structure comparison
    print("Creating table comparison diagram...")
    fig1 = create_table_comparison_diagram()
    fig1.savefig(output_dir / "table_comparison.png", dpi=300, bbox_inches='tight')
    fig1.savefig(output_dir / "table_comparison.pdf", bbox_inches='tight')
    print("✓ Saved: table_comparison.png/pdf")
    plt.close(fig1)
    
    # Diagram 2: Data flow
    print("Creating data flow diagram...")
    fig2 = create_data_flow_diagram()
    fig2.savefig(output_dir / "data_organization.png", dpi=300, bbox_inches='tight')
    fig2.savefig(output_dir / "data_organization.pdf", bbox_inches='tight')
    print("✓ Saved: data_organization.png/pdf")
    plt.close(fig2)
    
    print("\n" + "="*80)
    print("DIAGRAMS GENERATED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
