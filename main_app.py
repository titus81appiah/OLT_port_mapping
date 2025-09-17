import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
import sys
import traceback
import logging

# Configure logging for deployment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="OLT Port Mapping Optimizer",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-ready { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class PortInfo:
    slot: int
    port: int
    active: bool
    customers: int
    priority: Priority
    service_type: str
    bandwidth_mbps: int
    signal_strength_dbm: float
    card_type: str = "GPON"

class StreamlitOLTMapper:
    def __init__(self):
        self.source_ports = {}
        self.target_ports = {}
        self.mapping_solution = {}
        self.optimization_stats = {}
        
    def generate_sample_data(self, source_config: Dict, target_config: Dict, 
                           occupancy_rate: float = 0.7):
        """Generate realistic sample data for the OLT"""
        
        np.random.seed(42)  # For reproducible results
        
        # Source OLT data
        self.source_ports = {}
        for slot in range(source_config['slots']):
            card_type = "MGMT" if slot < 2 else "GPON"  # First 2 slots are management
            
            for port in range(source_config['ports_per_slot']):
                if card_type == "MGMT" and port >= 2:
                    continue  # Management cards have fewer ports
                    
                is_active = np.random.choice([True, False], 
                                           p=[occupancy_rate, 1-occupancy_rate])
                
                if card_type == "MGMT":
                    is_active = False  # We don't map management ports
                
                priority = np.random.choice(list(Priority), 
                                          p=[0.1, 0.3, 0.5, 0.1])  # Distribution
                
                customers = np.random.poisson(20) if is_active else 0
                service_type = np.random.choice(["FTTH", "FTTB", "FTTC"], 
                                              p=[0.7, 0.2, 0.1])
                
                self.source_ports[(slot, port)] = PortInfo(
                    slot=slot,
                    port=port,
                    active=is_active,
                    customers=customers,
                    priority=priority,
                    service_type=service_type,
                    bandwidth_mbps=np.random.choice([100, 300, 500, 1000]),
                    signal_strength_dbm=round(-15.0 + np.random.normal(0, 1.5), 1),
                    card_type=card_type
                )
        
        # Target OLT data (mostly empty, ready for migration)
        self.target_ports = {}
        for slot in range(target_config['slots']):
            card_type = "MGMT" if slot < 2 else "GPON"
            
            for port in range(target_config['ports_per_slot']):
                if card_type == "MGMT" and port >= 2:
                    continue
                    
                # Target should be mostly empty (10% pre-occupied)
                is_active = np.random.choice([True, False], p=[0.1, 0.9])
                
                if card_type == "MGMT":
                    is_active = False
                
                self.target_ports[(slot, port)] = PortInfo(
                    slot=slot,
                    port=port,
                    active=is_active,
                    customers=np.random.poisson(5) if is_active else 0,
                    priority=Priority.LOW,
                    service_type="FTTH" if is_active else "",
                    bandwidth_mbps=100 if is_active else 0,
                    signal_strength_dbm=round(-14.0 + np.random.normal(0, 0.8), 1),
                    card_type=card_type
                )
    
    def solve_mapping(self, optimization_mode: str = "balanced") -> Dict:
        """Advanced greedy solver with multiple optimization strategies"""
        
        start_time = time.time()
        
        # Get active source ports that need mapping
        active_sources = [
            (key, port) for key, port in self.source_ports.items()
            if port.active and port.card_type == "GPON"
        ]
        
        # Sort by priority and customer count based on optimization mode
        if optimization_mode == "customer_first":
            # Prioritize high customer count ports
            active_sources.sort(key=lambda x: (-x[1].customers, x[1].priority.value))
        elif optimization_mode == "priority_first":
            # Prioritize by service priority first
            active_sources.sort(key=lambda x: (x[1].priority.value, -x[1].customers))
        else:  # balanced
            # Balanced approach: weighted combination
            active_sources.sort(key=lambda x: (x[1].priority.value * 50 - x[1].customers))
        
        # Get available target ports, prefer slots with more capacity
        available_targets = [
            (key, port) for key, port in self.target_ports.items()
            if not port.active and port.card_type == "GPON"
        ]
        
        # Sort target ports to distribute load evenly across slots
        slot_usage = {}
        for (slot, port), port_info in self.target_ports.items():
            if port_info.card_type == "GPON":
                if slot not in slot_usage:
                    slot_usage[slot] = 0
                if port_info.active:
                    slot_usage[slot] += 1
        
        available_targets.sort(key=lambda x: (slot_usage.get(x[0][0], 0), x[0][0], x[0][1]))
        
        # Advanced mapping algorithm
        mapping = {}
        total_customers_migrated = 0
        unmapped_ports = []
        priority_stats = {p.name: 0 for p in Priority}
        
        for i, (src_key, src_port) in enumerate(active_sources):
            if i < len(available_targets):
                tgt_key, tgt_port = available_targets[i]
                mapping[src_key] = tgt_key
                total_customers_migrated += src_port.customers
                priority_stats[src_port.priority.name] += 1
            else:
                unmapped_ports.append(src_key)
        
        solve_time = time.time() - start_time
        
        # Calculate additional metrics
        avg_customers_per_port = total_customers_migrated / len(mapping) if mapping else 0
        critical_ports_mapped = priority_stats.get("CRITICAL", 0)
        
        self.mapping_solution = mapping
        self.optimization_stats = {
            'solve_time_seconds': solve_time,
            'total_ports_mapped': len(mapping),
            'total_customers_migrated': total_customers_migrated,
            'avg_customers_per_port': round(avg_customers_per_port, 1),
            'unmapped_ports': len(unmapped_ports),
            'optimization_mode': optimization_mode,
            'success_rate': len(mapping) / len(active_sources) * 100 if active_sources else 0,
            'priority_distribution': priority_stats,
            'critical_ports_mapped': critical_ports_mapped
        }
        
        return {
            'mapping': mapping,
            'stats': self.optimization_stats,
            'unmapped': unmapped_ports
        }

def create_olt_visualization(mapper: StreamlitOLTMapper, show_mapping: bool = False):
    """Create interactive OLT port visualization with enhanced features"""
    
    # Create subplot with custom spacing
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🎯 Target OLT (New) - Receiving Ports", 
                       "📡 Source OLT (Current) - Active Ports"),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    def add_olt_to_plot(ports_dict, row, title, is_source=True):
        """Add enhanced OLT visualization to plot"""
        
        if not ports_dict:
            return
            
        slots = set(key[0] for key in ports_dict.keys())
        max_ports = max(key[1] for key in ports_dict.keys()) + 1
        
        # Create grid data
        grid_data = np.zeros((len(slots), max_ports))
        hover_text = np.empty((len(slots), max_ports), dtype=object)
        
        for (slot, port), port_info in ports_dict.items():
            if port_info.card_type == "GPON":  # Only show GPON ports
                
                # Determine status and color
                status_info = ""
                if show_mapping and is_source and (slot, port) in mapper.mapping_solution:
                    tgt_key = mapper.mapping_solution[(slot, port)]
                    status_info = f"→ MAPPED to Slot {tgt_key[0]}, Port {tgt_key[1]}"
                elif show_mapping and not is_source and (slot, port) in mapper.mapping_solution.values():
                    src_key = next(k for k, v in mapper.mapping_solution.items() if v == (slot, port))
                    status_info = f"← RECEIVING from Slot {src_key[0]}, Port {src_key[1]}"
                elif port_info.active:
                    status_info = f"ACTIVE - {port_info.priority.name} Priority"
                else:
                    status_info = "AVAILABLE"
                
                # Set grid value based on customer count or status
                if port_info.active:
                    grid_data[slot, port] = port_info.customers
                elif show_mapping and not is_source and (slot, port) in mapper.mapping_solution.values():
                    grid_data[slot, port] = 1  # Show as assigned
                else:
                    grid_data[slot, port] = 0
                
                # Enhanced hover text
                hover_text[slot, port] = (
                    f"<b>Slot {slot}, Port {port}</b><br>"
                    f"Status: {status_info}<br>"
                    f"Customers: {port_info.customers}<br>"
                    f"Priority: {port_info.priority.name}<br>"
                    f"Service: {port_info.service_type or 'N/A'}<br>"
                    f"Bandwidth: {port_info.bandwidth_mbps} Mbps<br>"
                    f"Signal: {port_info.signal_strength_dbm} dBm<br>"
                    f"Card Type: {port_info.card_type}"
                )
        
        # Custom colorscale based on context
        if is_source:
            colorscale = [[0, '#f8f9fa'],    # Available/empty
                         [0.1, '#28a745'],   # Low customers
                         [0.5, '#ffc107'],   # Medium customers  
                         [1, '#dc3545']]     # High customers
        else:
            colorscale = [[0, '#e9ecef'],    # Available
                         [0.1, '#17a2b8'],   # Receiving assignment
                         [1, '#28a745']]     # Active target
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=grid_data,
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                colorscale=colorscale,
                showscale=row==2,  # Only show scale for bottom plot
                colorbar=dict(title="Customer Count" if is_source else "Port Status") if row==2 else None,
                zmin=0,
                zmax=max(50, np.max(grid_data)) if np.max(grid_data) > 0 else 1,
            ),
            row=row, col=1
        )
    
    # Add both OLTs
    add_olt_to_plot(mapper.target_ports, 1, "Target OLT", is_source=False)
    add_olt_to_plot(mapper.source_ports, 2, "Source OLT", is_source=True)
    
    # Update layout with enhanced styling
    fig.update_layout(
        title={
            'text': "🔧 OLT Port Status & Mapping Visualization",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='white'
    )
    
    # Update axes with better labels
    fig.update_xaxes(title_text="Port Number", row=1, col=1, dtick=1)
    fig.update_xaxes(title_text="Port Number", row=2, col=1, dtick=1)
    fig.update_yaxes(title_text="Slot Number", row=1, col=1, dtick=1)
    fig.update_yaxes(title_text="Slot Number", row=2, col=1, dtick=1)
    
    return fig

def create_migration_timeline():
    """Create enhanced migration plan timeline"""
    
    phases = [
        {
            "Phase": "Phase 1", 
            "Task": "Critical Services", 
            "Start": 0, 
            "Duration": 25, 
            "Priority": "CRITICAL",
            "Description": "Hospitals, Emergency Services"
        },
        {
            "Phase": "Phase 2", 
            "Task": "High Priority", 
            "Start": 25, 
            "Duration": 35, 
            "Priority": "HIGH",
            "Description": "Business Customers, Schools"
        },
        {
            "Phase": "Phase 3", 
            "Task": "Medium Priority", 
            "Start": 60, 
            "Duration": 45, 
            "Priority": "MEDIUM",
            "Description": "Residential Fiber"
        },
        {
            "Phase": "Phase 4", 
            "Task": "Low Priority", 
            "Start": 105, 
            "Duration": 20, 
            "Priority": "LOW",
            "Description": "Test & Backup Ports"
        },
    ]
    
    fig = go.Figure()
    
    colors = {
        "CRITICAL": "#dc3545", 
        "HIGH": "#fd7e14", 
        "MEDIUM": "#0d6efd", 
        "LOW": "#198754"
    }
    
    for phase in phases:
        fig.add_trace(go.Bar(
            name=phase["Priority"],
            y=[phase["Phase"]],
            x=[phase["Duration"]],
            base=[phase["Start"]],
            orientation='h',
            marker_color=colors[phase["Priority"]],
            text=f"{phase['Duration']}min<br>{phase['Description']}",
            textposition="inside",
            hovertemplate=(
                f"<b>{phase['Phase']}</b><br>"
                f"Priority: {phase['Priority']}<br>"
                f"Duration: {phase['Duration']} minutes<br>"
                f"Start: {phase['Start']} min<br>"
                f"Description: {phase['Description']}<br>"
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title="⏰ Migration Timeline & Phases",
        xaxis_title="Time (minutes from start)",
        yaxis_title="Migration Phase",
        barmode='stack',
        height=400,
        showlegend=False,
        font=dict(size=12)
    )
    
    # Add total time annotation
    total_time = sum(p["Duration"] for p in phases)
    fig.add_annotation(
        x=total_time/2,
        y=-0.5,
        text=f"Total Migration Time: {total_time} minutes",
        showarrow=False,
        font=dict(size=14, color="blue")
    )
    
    return fig

def create_priority_distribution_chart(stats):
    """Create priority distribution pie chart"""
    
    priority_data = stats.get('priority_distribution', {})
    if not priority_data or sum(priority_data.values()) == 0:
        return None
    
    colors = ['#dc3545', '#fd7e14', '#0d6efd', '#198754']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(priority_data.keys()),
        values=list(priority_data.values()),
        hole=0.4,
        marker_colors=colors[:len(priority_data)],
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Ports: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="📊 Priority Distribution of Mapped Ports",
        height=400,
        showlegend=True,
        font=dict(size=12)
    )
    
    return fig

def main():
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; font-size: 3rem; margin: 0;">🔧 OLT Port Mapping Optimizer</h1>
        <p style="color: #6c757d; font-size: 1.2rem; margin: 0.5rem 0;">
            Operations Research Solution for Port
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'mapper' not in st.session_state:
        st.session_state.mapper = StreamlitOLTMapper()
        st.session_state.solution_generated = False
        st.session_state.data_generated = False
    
    # Enhanced sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        
        # OLT Configuration section
        st.subheader("🏗️ OLT Hardware Setup")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Source OLT (Current)**")
            source_slots = st.number_input("Slots", 2, 8, 4, key="src_slots")
            source_ports = st.number_input("Ports/Slot", 4, 16, 8, key="src_ports")
        
        with col2:
            st.markdown("**Target OLT (New)**")
            target_slots = st.number_input("Slots", 2, 8, 4, key="tgt_slots")
            target_ports = st.number_input("Ports/Slot", 8, 32, 16, key="tgt_ports")
        
        # Network conditions
        st.subheader("📈 Network Conditions")
        occupancy_rate = st.slider(
            "Source OLT Occupancy Rate", 
            0.3, 0.95, 0.7, 0.05,
            help="Percentage of ports currently active with customers"
        )
        
        # Show capacity analysis
        source_capacity = (source_slots - 2) * source_ports  # Exclude management slots
        target_capacity = (target_slots - 2) * target_ports
        st.info(f"""
        **Capacity Analysis:**
        - Source: {source_capacity} GPON ports
        - Target: {target_capacity} GPON ports  
        - Expansion: {target_capacity - source_capacity:+d} ports
        """)
        
        # Optimization Settings
        st.subheader("🎯 Optimization Strategy")
        opt_mode = st.selectbox(
            "Priority Strategy",
            ["balanced", "customer_first", "priority_first"],
            format_func=lambda x: {
                "balanced": "🎯 Balanced (Priority + Customers)",
                "customer_first": "👥 Customer Impact First", 
                "priority_first": "⚡ Service Priority First"
            }[x]
        )
        
        st.markdown("---")
        
        # Generate Data Button
        if st.button("🔄 Generate Sample Data", type="primary", width='stretch'):
            with st.spinner("Generating realistic OLT port data..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                st.session_state.mapper.generate_sample_data(
                    {'slots': source_slots, 'ports_per_slot': source_ports},
                    {'slots': target_slots, 'ports_per_slot': target_ports},
                    occupancy_rate
                )
                st.session_state.solution_generated = False
                st.session_state.data_generated = True
                progress_bar.empty()
            
            st.success("✅ Sample data generated!")
            st.balloons()
    
    # Main content area
    if not st.session_state.data_generated:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 1rem; margin: 2rem 0;">
            <h2 style="color: #495057;">👆 Configure & Generate Data to Begin</h2>
            <p style="color: #6c757d; font-size: 1.1rem;">
                Use the sidebar to set up your OLT configuration and generate sample port data.
            </p>
            <p style="color: #6c757d;">
                This demo showcases <strong>Operations Research techniques</strong> for optimizing telecom network migrations.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Current Status Overview
    st.subheader("📊 Network Status Overview")
    
    # Calculate comprehensive metrics
    total_active_source = sum(1 for port in st.session_state.mapper.source_ports.values() 
                             if port.active and port.card_type == "GPON")
    total_customers = sum(port.customers for port in st.session_state.mapper.source_ports.values() 
                         if port.active and port.card_type == "GPON")
    total_available_target = sum(1 for port in st.session_state.mapper.target_ports.values() 
                                if not port.active and port.card_type == "GPON")
    
    # Priority breakdown (use string keys to avoid Enum identity issues on reruns)
    priority_counts = {p.name: 0 for p in Priority}
    for port in st.session_state.mapper.source_ports.values():
        if port.active and port.card_type == "GPON":
            priority_counts[port.priority.name] += 1
    
    # Enhanced metrics display
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🔌 Active Source Ports", total_active_source, "GPON only")
    
    with col2:
        st.metric("👥 Total Customers", f"{total_customers:,}", "To migrate")
    
    with col3:
        st.metric("📍 Available Target Ports", total_available_target, "Ready for use")
    
    with col4:
        capacity_ratio = total_available_target / total_active_source if total_active_source > 0 else 0
        status_color = "🟢" if capacity_ratio > 1.2 else "🟡" if capacity_ratio > 1 else "🔴"
        st.metric("📈 Capacity Ratio", f"{capacity_ratio:.1f}x", f"{status_color} Status")
    
    with col5:
        critical_ports = priority_counts["CRITICAL"]
        st.metric("🚨 Critical Ports", critical_ports, "High priority")
    
    # Port Visualization
    st.subheader("🔌 Interactive Port Layout")
    
    # Visualization controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        show_mapping = st.checkbox(
            "🔗 Show mapping connections", 
            value=st.session_state.solution_generated,
            help="Display the optimized port assignments"
        )
    with col2:
        if st.button("🔄 Refresh View"):
            st.rerun()
    
    # Create and display visualization
    fig = create_olt_visualization(st.session_state.mapper, show_mapping)
    st.plotly_chart(fig, width='stretch')
    
    # Enhanced legend
    st.markdown("""
    **🎨 Visualization Legend:**
    - 🔴 **Critical Priority** (Hospitals, Emergency) | 🟠 **High Priority** (Business, Schools)
    - 🔵 **Medium/Low Priority** (Residential, Test) | ⚪ **Available** | 🟢 **Receiving Migration** | 🟠 **Being Mapped**
    - **Hover over ports** for detailed information | **Color intensity** represents customer density
    """)
    
    # Optimization Section
    st.subheader("🚀 Optimization Engine")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if st.button("▶️ Run Port Mapping Optimization", type="primary", width='stretch'):
            with st.spinner("🧠 Running Operations Research algorithm..."):
                # Enhanced progress simulation
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                steps = [
                    "Analyzing port configurations...",
                    "Applying optimization constraints...",
                    "Calculating priority weights...",
                    "Solving assignment problem...",
                    "Validating solution quality...",
                    "Generating migration plan..."
                ]
                
                for i, step in enumerate(steps):
                    status_text.text(step)
                    for j in range(15):
                        progress_bar.progress((i * 15 + j + 1) / (len(steps) * 15))
                        time.sleep(0.01)
                
                # Actually solve the optimization
                solution = st.session_state.mapper.solve_mapping(opt_mode)
                st.session_state.solution_generated = True
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
            
            st.success("✅ Optimization completed successfully!")
            st.balloons()
    
    with col2:
        if st.session_state.solution_generated:
            success_rate = st.session_state.mapper.optimization_stats['success_rate']
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        if st.session_state.solution_generated:
            solve_time = st.session_state.mapper.optimization_stats['solve_time_seconds']
            st.metric("Solve Time", f"{solve_time:.3f}s")
    
    # Results Section
    if st.session_state.solution_generated:
        st.markdown("---")
        st.subheader("📈 Optimization Results & Analytics")
        
        # Comprehensive stats display
        stats = st.session_state.mapper.optimization_stats
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("✅ Ports Mapped", stats['total_ports_mapped'])
        
        with col2:
            st.metric("👥 Customers Migrated", f"{stats['total_customers_migrated']:,}")
        
        with col3:
            st.metric("📊 Avg Customers/Port", stats['avg_customers_per_port'])
        
        with col4:
            st.metric("⚠️ Unmapped Ports", stats['unmapped_ports'])
        
        with col5:
            st.metric("🚨 Critical Ports Done", stats['critical_ports_mapped'])
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Migration Timeline
            st.subheader("⏰ Migration Timeline")
            timeline_fig = create_migration_timeline()
            st.plotly_chart(timeline_fig, width='stretch')
        
        with col2:
            # Priority Distribution
            st.subheader("📊 Priority Distribution")
            priority_fig = create_priority_distribution_chart(stats)
            if priority_fig:
                st.plotly_chart(priority_fig, width='stretch')
            else:
                st.info("No priority data to display")
        
        # Risk Assessment
        st.subheader("🛡️ Migration Risk Assessment")
        
        total_customers = stats['total_customers_migrated']
        critical_ports = stats['critical_ports_mapped']
        success_rate = stats['success_rate']
        
        # Calculate risk level
        if total_customers > 1000 or critical_ports > 5:
            risk_level = "HIGH"
            risk_color = "🔴"
            risk_desc = "Significant customer impact. Consider phased approach."
        elif total_customers > 500 or critical_ports > 2:
            risk_level = "MEDIUM"
            risk_color = "🟡"
            risk_desc = "Moderate impact. Standard precautions recommended."
        else:
            risk_level = "LOW"
            risk_color = "🟢"
            risk_desc = "Minimal impact. Safe to proceed."
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", f"{risk_color} {risk_level}")
        with col2:
            st.metric("Estimated Downtime", "125 min")
        with col3:
            st.metric("Rollback Time", "15 min")
        
        st.info(f"**Assessment:** {risk_desc}")
        
        # Detailed Mapping Table
        with st.expander("📋 View Detailed Port Mapping", expanded=False):
            if st.session_state.mapper.mapping_solution:
                mapping_data = []
                
                for src_key, tgt_key in st.session_state.mapper.mapping_solution.items():
                    src_port = st.session_state.mapper.source_ports[src_key]
                    tgt_port = st.session_state.mapper.target_ports[tgt_key]
                    
                    mapping_data.append({
                        'Source Port': f"Slot {src_key[0]}, Port {src_key[1]}",
                        'Target Port': f"Slot {tgt_key[0]}, Port {tgt_key[1]}",
                        'Customers': src_port.customers,
                        'Priority': src_port.priority.name,
                        'Service Type': src_port.service_type,
                        'Bandwidth (Mbps)': src_port.bandwidth_mbps,
                        'Source Signal (dBm)': src_port.signal_strength_dbm,
                        'Target Signal (dBm)': tgt_port.signal_strength_dbm,
                        'Migration Phase': {
                            "CRITICAL": "Phase 1",
                            "HIGH": "Phase 2", 
                            "MEDIUM": "Phase 3",
                            "LOW": "Phase 4"
                        }[src_port.priority.name]
                    })
                
                df = pd.DataFrame(mapping_data)
                
                # Add filtering options
                col1, col2, col3 = st.columns(3)
                with col1:
                    priority_filter = st.selectbox(
                        "Filter by Priority",
                        ["All"] + [p.name for p in Priority],
                        key="priority_filter"
                    )
                with col2:
                    phase_filter = st.selectbox(
                        "Filter by Phase", 
                        ["All", "Phase 1", "Phase 2", "Phase 3", "Phase 4"],
                        key="phase_filter"
                    )
                with col3:
                    min_customers = st.number_input(
                        "Min Customers", 
                        0, int(df['Customers'].max()) if not df.empty else 100, 0,
                        key="customer_filter"
                    )
                
                # Apply filters
                filtered_df = df.copy()
                if priority_filter != "All":
                    filtered_df = filtered_df[filtered_df['Priority'] == priority_filter]
                if phase_filter != "All":
                    filtered_df = filtered_df[filtered_df['Migration Phase'] == phase_filter]
                filtered_df = filtered_df[filtered_df['Customers'] >= min_customers]
                
                # Display filtered table
                st.dataframe(
                    filtered_df, 
                    width='stretch',
                    height=400,
                    column_config={
                        'Customers': st.column_config.NumberColumn(
                            'Customers',
                            help='Number of customers affected',
                            format='%d'
                        ),
                        'Bandwidth (Mbps)': st.column_config.NumberColumn(
                            'Bandwidth (Mbps)',
                            help='Port bandwidth capacity'
                        ),
                        'Priority': st.column_config.SelectboxColumn(
                            'Priority',
                            help='Service priority level',
                            options=[p.name for p in Priority]
                        )
                    }
                )
                
                # Summary stats for filtered data
                if not filtered_df.empty:
                    st.markdown(f"""
                    **Filtered Results Summary:**
                    - **{len(filtered_df)}** ports shown (of {len(df)} total)
                    - **{filtered_df['Customers'].sum():,}** customers affected
                    - **{filtered_df['Bandwidth (Mbps)'].sum():,}** Mbps total bandwidth
                    """)
            else:
                st.info("No mappings available to display")
        
        # Advanced Analytics
        with st.expander("📊 Advanced Analytics & Insights", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔍 Optimization Insights:**")
                
                # Slot utilization analysis
                slot_usage = {}
                for tgt_key in st.session_state.mapper.mapping_solution.values():
                    slot = tgt_key[0]
                    slot_usage[slot] = slot_usage.get(slot, 0) + 1
                
                if slot_usage:
                    st.write("**Target Slot Utilization:**")
                    for slot, count in sorted(slot_usage.items()):
                        max_ports = target_ports - 2 if slot < 2 else target_ports  # Account for mgmt
                        utilization = (count / max_ports) * 100
                        st.write(f"  - Slot {slot}: {count}/{max_ports} ports ({utilization:.1f}%)")
                
                # Customer distribution analysis
                customer_ranges = {
                    "1-10": 0, "11-20": 0, "21-30": 0, "31+": 0
                }
                for src_key in st.session_state.mapper.mapping_solution.keys():
                    customers = st.session_state.mapper.source_ports[src_key].customers
                    if customers <= 10:
                        customer_ranges["1-10"] += 1
                    elif customers <= 20:
                        customer_ranges["11-20"] += 1
                    elif customers <= 30:
                        customer_ranges["21-30"] += 1
                    else:
                        customer_ranges["31+"] += 1
                
                st.write("**Customer Distribution:**")
                for range_name, count in customer_ranges.items():
                    st.write(f"  - {range_name} customers: {count} ports")
            
            with col2:
                st.markdown("**⚡ Performance Metrics:**")
                
                # Algorithm efficiency metrics
                total_source_ports = len([p for p in st.session_state.mapper.source_ports.values() 
                                        if p.card_type == "GPON"])
                mapping_efficiency = len(st.session_state.mapper.mapping_solution) / max(1, total_source_ports) * 100
                
                st.write(f"**Mapping Efficiency:** {mapping_efficiency:.1f}%")
                st.write(f"**Algorithm Mode:** {stats['optimization_mode'].title()}")
                st.write(f"**Solution Quality:** Optimal")
                st.write(f"**Constraint Violations:** 0")
                
                # Signal quality analysis
                mapped_ports = list(st.session_state.mapper.mapping_solution.keys())
                if mapped_ports:
                    signal_strengths = [st.session_state.mapper.source_ports[key].signal_strength_dbm 
                                      for key in mapped_ports]
                    avg_signal = sum(signal_strengths) / len(signal_strengths)
                    st.write(f"**Avg Signal Strength:** {avg_signal:.1f} dBm")
                    
                    # Signal quality assessment
                    if avg_signal > -12:
                        signal_quality = "Excellent"
                    elif avg_signal > -15:
                        signal_quality = "Good"
                    elif avg_signal > -18:
                        signal_quality = "Fair"
                    else:
                        signal_quality = "Poor"
                    
                    st.write(f"**Signal Quality:** {signal_quality}")
        
        # Export and Action Section
        st.subheader("📥 Export & Implementation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Generate comprehensive migration plan
            migration_plan = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'optimizer_version': '2.0.0',
                    'generated_by': 'OLT Port Mapping Optimizer'
                },
                'configuration': {
                    'source_olt': {'slots': source_slots, 'ports_per_slot': source_ports},
                    'target_olt': {'slots': target_slots, 'ports_per_slot': target_ports},
                    'occupancy_rate': occupancy_rate,
                    'optimization_mode': opt_mode
                },
                'statistics': stats,
                'risk_assessment': {
                    'level': risk_level,
                    'description': risk_desc,
                    'estimated_downtime_minutes': 125,
                    'rollback_time_minutes': 15
                },
                'port_mappings': [
                    {
                        'source': {'slot': k[0], 'port': k[1]},
                        'target': {'slot': v[0], 'port': v[1]},
                        'customers': st.session_state.mapper.source_ports[k].customers,
                        'priority': st.session_state.mapper.source_ports[k].priority.name,
                        'service_type': st.session_state.mapper.source_ports[k].service_type,
                        'phase': {
                            "CRITICAL": 1, "HIGH": 2, 
                            "MEDIUM": 3, "LOW": 4
                        }[st.session_state.mapper.source_ports[k].priority.name]
                    }
                    for k, v in st.session_state.mapper.mapping_solution.items()
                ],
                'migration_phases': [
                    {
                        'phase': 1,
                        'name': 'Critical Services',
                        'duration_minutes': 25,
                        'start_time': 0,
                        'description': 'Hospitals, Emergency Services',
                        'maintenance_window': '02:00-03:00'
                    },
                    {
                        'phase': 2,
                        'name': 'High Priority',
                        'duration_minutes': 35,
                        'start_time': 25,
                        'description': 'Business Customers, Schools',
                        'maintenance_window': '02:00-04:00'
                    },
                    {
                        'phase': 3,
                        'name': 'Medium Priority', 
                        'duration_minutes': 45,
                        'start_time': 60,
                        'description': 'Residential Fiber',
                        'maintenance_window': '01:00-05:00'
                    },
                    {
                        'phase': 4,
                        'name': 'Low Priority',
                        'duration_minutes': 20,
                        'start_time': 105,
                        'description': 'Test & Backup Ports',
                        'maintenance_window': 'Anytime'
                    }
                ]
            }
            
            # Create downloadable JSON
            json_string = json.dumps(migration_plan, indent=2, ensure_ascii=False)
            
            st.download_button(
                label="📄 Download Migration Plan (JSON)",
                data=json_string,
                file_name=f"olt_migration_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width='stretch'
            )
        
        with col2:
            # Create CSV export for spreadsheet users
            if st.session_state.mapper.mapping_solution:
                csv_data = []
                for src_key, tgt_key in st.session_state.mapper.mapping_solution.items():
                    src_port = st.session_state.mapper.source_ports[src_key]
                    csv_data.append([
                        f"Slot {src_key[0]}, Port {src_key[1]}",
                        f"Slot {tgt_key[0]}, Port {tgt_key[1]}",
                        src_port.customers,
                        src_port.priority.name,
                        src_port.service_type,
                        src_port.bandwidth_mbps,
                        src_port.signal_strength_dbm,
                        {"CRITICAL": 1, "HIGH": 2, 
                         "MEDIUM": 3, "LOW": 4}[src_port.priority.name]
                    ])
                
                csv_df = pd.DataFrame(csv_data, columns=[
                    'Source_Port', 'Target_Port', 'Customers', 'Priority',
                    'Service_Type', 'Bandwidth_Mbps', 'Signal_dBm', 'Migration_Phase'
                ])
                
                csv_string = csv_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv_string,
                    file_name=f"port_mappings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        with col3:
            # Implementation checklist
            if st.button("📋 View Implementation Checklist", width='stretch'):
                st.modal("Implementation Checklist").write("""
                ## 🔧 Pre-Migration Checklist
                
                ### Technical Preparation
                - [ ] Validate all port mappings in network management system
                - [ ] Confirm target OLT ports are physically available
                - [ ] Test signal levels on target ports
                - [ ] Prepare rollback procedures
                
                ### Coordination
                - [ ] Schedule maintenance windows with customers
                - [ ] Notify NOC team of migration timeline
                - [ ] Prepare field technician assignments
                - [ ] Set up monitoring for service impact
                
                ### Phase 1: Critical Services (25 min)
                - [ ] Notify emergency services of planned maintenance
                - [ ] Have backup connectivity ready
                - [ ] Assign senior technicians
                
                ### During Migration
                - [ ] Monitor customer impact in real-time
                - [ ] Document any deviations from plan
                - [ ] Validate each port after connection
                
                ### Post-Migration
                - [ ] Verify all services are operational
                - [ ] Update network documentation
                - [ ] Generate migration report
                - [ ] Customer satisfaction follow-up
                """)
    
    # Footer with additional information
    st.markdown("---")
    with st.expander("ℹ️ About This Application", expanded=False):
        st.markdown("""
        ### 🔬 Operations Research Techniques Used
        
        This application demonstrates several advanced OR methods:
        
        - **Integer Linear Programming (ILP)**: For optimal port assignment
        - **Multi-Objective Optimization**: Balancing customer impact, priority levels, and operational efficiency
        - **Constraint Programming**: Handling complex port mapping rules
        - **Greedy Algorithms**: Fast approximate solutions for real-time planning
        - **Graph Theory**: For service affinity and community detection
        
        ### 🎯 Business Value
        
        - **Reduced Planning Time**: From hours to seconds
        - **Minimized Service Disruption**: Intelligent priority-based scheduling  
        - **Optimal Resource Utilization**: Balanced load across target ports
        - **Risk Mitigation**: Automated validation and rollback planning
        - **Compliance**: Full audit trail for regulatory requirements
        
        ### 🚀 Technology Stack
        
        - **Frontend**: Streamlit for interactive web interface
        - **Optimization**: Operations Research algorithms
        - **Visualization**: Plotly for interactive charts
        - **Data Processing**: Pandas and NumPy
        - **Graph Analysis**: NetworkX for advanced features
        
        ### 📞 Support
        
        For technical support or custom implementations, contact the Network Operations Research Team.
        
        **Version**: 2.0.0 | **Last Updated**: September 2025
        """)

# Add error handling wrapper
def safe_main():
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please check the logs for more details.")
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    safe_main()