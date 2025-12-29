"""
Streamlit web application for searching houses using natural language queries.
"""

import streamlit as st
import sys
import requests
import json
import streamlit.components.v1 as components
from pathlib import Path

# Add parent directory to path to import common modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after streamlit is available to access secrets
from common.search import search_houses
from common.config import settings

# Reload settings to pick up Streamlit secrets (if available)
# This needs to happen after streamlit is imported
try:
    # Update settings from Streamlit secrets if they exist
    if hasattr(st, 'secrets'):
        # Direct access to secrets - Streamlit Cloud stores them as top-level keys
        try:
            settings.pinecone_api_key = str(st.secrets["PINECONE_API_KEY"])
        except (KeyError, TypeError, AttributeError):
            pass
        
        try:
            settings.pinecone_index = str(st.secrets["PINECONE_INDEX"])
        except (KeyError, TypeError, AttributeError):
            pass
        
        try:
            settings.model_name = str(st.secrets["MODEL_NAME"])
        except (KeyError, TypeError, AttributeError):
            pass
        
        try:
            settings.default_top_k = int(st.secrets["DEFAULT_TOP_K"])
        except (KeyError, TypeError, AttributeError, ValueError):
            pass
except (AttributeError, RuntimeError):
    # Streamlit not available or not in Streamlit context
    pass


def format_price(price: float) -> str:
    """Format price as currency string."""
    return f"${int(price):,}"


def build_filters(
    min_bedrooms: int,
    max_bedrooms: int,
    min_bathrooms: float,
    max_bathrooms: float,
    min_price: float,
    max_price: float,
    min_area: int,
    max_area: int,
    zipcode: str
) -> dict | None:
    """Build Pinecone filter dictionary from UI inputs."""
    filters = {}
    
    # Bedroom filters
    if min_bedrooms > 0:
        filters["bedrooms"] = {"$gte": min_bedrooms}
    if max_bedrooms > 0:
        if "bedrooms" in filters:
            filters["bedrooms"]["$lte"] = max_bedrooms
        else:
            filters["bedrooms"] = {"$lte": max_bedrooms}
    
    # Bathroom filters
    if min_bathrooms > 0:
        filters["bathrooms"] = {"$gte": min_bathrooms}
    if max_bathrooms > 0:
        if "bathrooms" in filters:
            filters["bathrooms"]["$lte"] = max_bathrooms
        else:
            filters["bathrooms"] = {"$lte": max_bathrooms}
    
    # Price filters
    if min_price > 0:
        filters["price"] = {"$gte": min_price}
    if max_price > 0:
        if "price" in filters:
            filters["price"]["$lte"] = max_price
        else:
            filters["price"] = {"$lte": max_price}
    
    # Area filters
    if min_area > 0:
        filters["area"] = {"$gte": min_area}
    if max_area > 0:
        if "area" in filters:
            filters["area"]["$lte"] = max_area
        else:
            filters["area"] = {"$lte": max_area}
    
    # Zipcode filter
    if zipcode:
        filters["zipcode"] = {"$eq": str(zipcode)}
    
    return filters if filters else None


def display_house_images(images, base_path: str = None):
    """
    Display house images if available.
    
    Args:
        images: Dictionary of images or JSON string, or None
        base_path: Base path for resolving image paths (defaults to project root)
    """
    if not images:
        return
    
    # Handle JSON string (from Pinecone metadata)
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except json.JSONDecodeError:
            st.caption("⚠️ Images data format error")
            return
    
    if not isinstance(images, dict):
        return
    
    # Set base path to project root if not provided
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    image_types = ["frontal", "bedroom", "kitchen", "bathroom"]
    available_images = []
    image_paths_info = []
    
    # Try demo_images first (for Streamlit Cloud), then fall back to full dataset
    demo_images_base = Path(base_path) / "data" / "Houses-dataset" / "demo_images"
    full_images_base = Path(base_path) / "data" / "Houses-dataset" / "Houses Dataset"
    
    for img_type in image_types:
        if img_type in images:
            img_path = images[img_type]
            if not img_path:
                continue
            
            # Extract filename from path (handles paths like "Houses-dataset/Houses Dataset/1_bathroom.jpg")
            filename = Path(img_path).name
            
            # Store info about the image
            image_paths_info.append((img_type, filename, img_path))
            
            # Check demo_images first (for Streamlit Cloud), then full dataset
            demo_path = demo_images_base / filename
            full_path = full_images_base / filename
            
            # Debug: Check both paths
            if demo_path.exists() and demo_path.is_file():
                available_images.append((img_type, str(demo_path)))
            elif full_path.exists() and full_path.is_file():
                available_images.append((img_type, str(full_path)))
            # If neither exists, we'll show the message below
    
    if available_images:
        # Display images in a grid
        num_cols = min(len(available_images), 4)
        cols = st.columns(num_cols)
        for idx, (img_type, img_path) in enumerate(available_images):
            with cols[idx % num_cols]:
                try:
                    st.image(
                        img_path,
                        caption=img_type.capitalize(),
                        use_container_width=True
                    )
                except Exception as e:
                    st.caption(f"Could not load {img_type} image: {str(e)}")
    elif image_paths_info:
        # Images are referenced but files don't exist
        # Show which images would be available and debug info
        image_types_found = [img_type.capitalize() for img_type, _, _ in image_paths_info]
        filename_example = image_paths_info[0][1] if image_paths_info else "N/A"
        
        # Debug: Check if directories exist
        demo_exists = demo_images_base.exists()
        full_exists = full_images_base.exists()
        
        debug_info = f" (demo_dir exists: {demo_exists}, full_dir exists: {full_exists})"
        st.caption(f"📷 Images referenced: {', '.join(image_types_found)}. *Image files not found. Looking for: {filename_example}*{debug_info}")


# Page configuration
st.set_page_config(
    page_title="MLLM House Search",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .house-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        background-color: #667eea;
        color: white;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏠 MLLM House Search</h1>', unsafe_allow_html=True)
#st.markdown("### Find your dream home using natural language")

# Initialize session state for auto-search
if "auto_search" not in st.session_state:
    st.session_state.auto_search = False

# API Configuration (optional - can use direct search or API)
use_api = st.sidebar.checkbox("Use FastAPI Backend", value=False, help="Toggle to use FastAPI backend instead of direct search")
api_url = st.sidebar.text_input("API URL", value="http://localhost:8000")

# Check API availability if enabled
api_available = False
if use_api:
    try:
        response = requests.get(f"{api_url}/health", timeout=2)
        if response.status_code == 200:
            api_available = True
        else:
            api_available = False
    except (requests.exceptions.RequestException, Exception):
        api_available = False
        if use_api:
            st.sidebar.warning("⚠️ FastAPI backend is not available. Make sure the server is running:\n\n```bash\npython app/api.py\n# or\nuvicorn app.api:app --reload\n```")
            st.sidebar.info("The app will automatically use direct search instead.")

# Query input with examples
st.markdown("---")

# Initialize query in session state if not exists
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

query = st.text_input(
    "**Search:**",
    value=st.session_state.current_query,
    placeholder="e.g., Modern 3-bedroom house with a large kitchen and backyard",
    help="Use natural language to describe what you're looking for",
    key="query_input"
)

# Update session state when query changes manually
if query != st.session_state.current_query:
    st.session_state.current_query = query

# Example queries
with st.expander("💡 Example Queries", expanded=False):
    examples = [
        "Modern 3-bedroom house with a large kitchen and backyard",
        "Home with good natural lighting and updated appliances",
    ]
    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{example}", use_container_width=True):
            # When an example is clicked, update the query *and* trigger auto search
            st.session_state.current_query = example
            st.session_state.auto_search = True  # <-- no st.rerun needed anymore

# Use session state query for search logic *after* examples, so it sees example clicks
search_query = st.session_state.current_query if st.session_state.current_query else query

# Filters section
with st.expander("🔍 Filters (Optional)", expanded=False):
    st.caption("Set to 0 to disable a filter")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Bedrooms**")
        min_beds = st.number_input("Min", 0, 10, 0, key="min_bedrooms")
        max_beds = st.number_input("Max", 0, 10, 0, key="max_bedrooms")
    
    with col2:
        st.markdown("**Bathrooms**")
        min_baths = st.number_input("Min", 0.0, 10.0, 0.0, step=0.5, key="min_bathrooms")
        max_baths = st.number_input("Max", 0.0, 10.0, 0.0, step=0.5, key="max_bathrooms")
    
    with col3:
        st.markdown("**Price ($)**")
        min_price = st.number_input("Min", 0, 2_000_000, 0, step=50000, key="min_price")
        max_price = st.number_input("Max", 0, 2_000_000, 0, step=50000, key="max_price")
    
    with col4:
        st.markdown("**Area (sqft)**")
        min_area = st.number_input("Min", 0, 10000, 0, step=100, key="min_area")
        max_area = st.number_input("Max", 0, 10000, 0, step=100, key="max_area")
    
    zipcode = st.text_input("Zipcode", "", placeholder="e.g., 85255")

# Search settings
col1, col2 = st.columns([3, 1])
with col1:
    top_k = st.slider("Number of results", 1, 20, 10)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing

# Search button and results
st.markdown("---")
search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

# Trigger search if example query was clicked or search button was clicked
should_search = search_clicked or st.session_state.auto_search
search_query_to_use = search_query if search_query else query

if should_search and search_query_to_use and search_query_to_use.strip():
    # 1. Capture if this was an auto_search BEFORE resetting the flag
    #    We need this to decide whether to auto-scroll later
    was_auto_search = st.session_state.auto_search
    
    # Reset auto_search flag after we've determined we should search
    if st.session_state.auto_search:
        st.session_state.auto_search = False
    
    # Use the search query (from session state or input)
    query = search_query_to_use
    
    if query:
        # Build filters
        filters = build_filters(
            min_bedrooms=min_beds,
            max_bedrooms=max_beds,
            min_bathrooms=min_baths,
            max_bathrooms=max_baths,
            min_price=min_price,
            max_price=max_price,
            min_area=min_area,
            max_area=max_area,
            zipcode=zipcode
        )
        
        # Show active filters
        if filters:
            with st.expander("📋 Active Filters", expanded=False):
                st.json(filters)
        
        # Perform search
        with st.spinner("🔍 Searching..."):
            try:
                # Only use API if it's enabled AND available
                if use_api and api_available:
                    # Use FastAPI backend
                    try:
                        response = requests.post(
                            f"{api_url}/api/search",
                            json={
                                "query": query,
                                "top_k": top_k,
                                "filters": {
                                    "min_bedrooms": min_beds if min_beds > 0 else None,
                                    "max_bedrooms": max_beds if max_beds > 0 else None,
                                    "min_bathrooms": min_baths if min_baths > 0 else None,
                                    "max_bathrooms": max_baths if max_baths > 0 else None,
                                    "min_price": min_price if min_price > 0 else None,
                                    "max_price": max_price if max_price > 0 else None,
                                    "min_area": min_area if min_area > 0 else None,
                                    "max_area": max_area if max_area > 0 else None,
                                    "zipcode": zipcode if zipcode else None
                                } if filters else None
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        data = response.json()
                        matches = data["results"]
                    except requests.exceptions.RequestException as e:
                        # If API fails during request, try to extract error details
                        error_detail = str(e)
                        error_full = str(e)
                        if hasattr(e, 'response') and e.response is not None:
                            try:
                                error_json = e.response.json()
                                if 'detail' in error_json:
                                    error_detail = error_json['detail']
                                    error_full = error_json['detail']
                            except:
                                pass
                        
                        # If API fails during request, fall back to direct search
                        st.warning(f"⚠️ API request failed: {error_detail[:200]}...")
                        with st.expander("View Full Error Details"):
                            st.code(error_full, language="text")
                        st.info("🔄 Falling back to direct search...")
                        matches = search_houses(query, top_k=top_k, filters=filters)
                else:
                    # Use direct search (either API not enabled or not available)
                    if use_api and not api_available:
                        st.info("ℹ️ Using direct search (FastAPI backend not available)")
                    matches = search_houses(query, top_k=top_k, filters=filters)
                
                if matches:
                    # 2. Add an invisible HTML anchor here
                    st.markdown('<div id="results_anchor"></div>', unsafe_allow_html=True)
                    
                    st.success(f"✅ Found {len(matches)} result(s) for: *{query}*")
                    
                    # 3. Inject JavaScript to scroll to the anchor IF it was an auto-search
                    if was_auto_search:
                        js_code = """
                            <script>
                                var element = window.parent.document.getElementById("results_anchor");
                                if (element) {
                                    element.scrollIntoView({behavior: "smooth", block: "start"});
                                }
                            </script>
                        """
                        components.html(js_code, height=0, width=0)

                    st.markdown("---")
                    
                    # Display results in cards
                    for idx, m in enumerate(matches, 1):
                        # Handle both dict-like and object-like matches
                        if hasattr(m, 'metadata'):
                            meta = m.metadata
                            house_id = m.id
                            score = m.score
                        else:
                            meta = m.get("metadata", {})
                            house_id = m.get("id", "N/A")
                            score = m.get("score", 0.0)
                        
                        # Create a card-like container
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"### 🏠 House #{house_id}")
                            with col2:
                                # Similarity score badge
                                score_percent = int(score * 100)
                                st.markdown(
                                    f'<div class="similarity-badge">Match: {score_percent}%</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Basic info in columns
                            bedrooms = int(meta.get("bedrooms", 0))
                            bathrooms = float(meta.get("bathrooms", 0))
                            area = int(meta.get("area", 0))
                            price = float(meta.get("price", 0))
                            zipcode_val = meta.get("zipcode", "N/A")
                            
                            info_cols = st.columns(5)
                            with info_cols[0]:
                                st.metric("Bedrooms", bedrooms)
                            with info_cols[1]:
                                st.metric("Bathrooms", bathrooms)
                            with info_cols[2]:
                                st.metric("Area", f"{area:,} sqft")
                            with info_cols[3]:
                                st.metric("Price", format_price(price))
                            with info_cols[4]:
                                st.metric("Zipcode", zipcode_val)
                            
                            # Images if available
                            if "images" in meta and meta["images"]:
                                st.markdown("**Images:**")
                                display_house_images(meta["images"])
                                st.markdown("")  # spacing
                            
                            # Description
                            description = meta.get("description", "No description available")
                            st.markdown("**Description:**")
                            st.markdown(f"*{description}*")
                            
                            # Divider
                            st.markdown("---")
                        
                else:
                    st.warning("⚠️ No results found. Try adjusting your search query or filters.")
                    
            except Exception as e:
                st.error(f"❌ Error during search: {str(e)}")
                with st.expander("Error Details"):
                    st.exception(e)
    else:
        st.info("Please enter a search query to find houses. Try: 'Modern 3-bedroom house with a large kitchen and backyard'")
# Sidebar with info
with st.sidebar:
    st.markdown("---")
    st.markdown("ℹ️ About")
    st.markdown("""
    This demo uses semantic search powered by vector embeddings to find houses 
    based on natural language descriptions. You can combine hard filters with natural language 
    queries for more precise results.
    """)
 
    st.markdown("---")
    st.header("📊 Details")
    st.markdown(f"""
    - **Model**: Sentence Transformers
    - **Vector DB**: Pinecone
    - **Search Type**: Semantic similarity
    - **Results**: Top-K nearest neighbors
    - **Credits**: Nick Sheft
    """)
