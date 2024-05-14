import streamlit as st
from pinecone import Pinecone
from utils import get_text_embedding
from image_embeddings import get_image_embedding


PINECONE_KEY = "508a1fea-8fd8-4b51-ae51-053df59dd9a7"
st.set_page_config(layout="wide")
st.markdown(
    """
<style>
#main-header {
        font-size: 2.5rem;
        color: #1f1f1f;
        text-align: center;
        padding: 20px 0;
    }
#introduction {
    padding: 0px 20px 0px 20px;
    background-color: #ffffd9;
    border-radius: 10px;

}
#introduction p {
    font-size: 1.1rem;
    color: #a17112;

}
img {
    padding: 5px;
}
</style>


""",
    unsafe_allow_html=True,
)

st.markdown("<div id='main-header'>CLIP Search Engine with Pinecone Database</div>", unsafe_allow_html=True)

st.markdown(
    """
<div id="introduction">

<p>
This is a demo for image engine search powered by CLIP model and Pinecone database.
The image search can be conducted by querying through texts or images. Enjoy it :3
</p>
</div>
""",
    unsafe_allow_html=True,
)


pc = Pinecone(api_key="508a1fea-8fd8-4b51-ae51-053df59dd9a7")
index = pc.Index("clip-image-search")

#query by text
text_query = st.text_input(":mag_right: Search for images by text", "salad on a plate")
number_of_results = st.slider("Number of results ", 1, 100, 10, key='text')

if text_query is not None:
    query_vector_text = get_text_embedding(text_query)
    top_k_samples = index.query(
       vector=query_vector_text, top_k=number_of_results, include_values=False
    )

st.markdown("<div style='align: center; display: flex'>", unsafe_allow_html=True)
st.image([str(result.id) for result in top_k_samples["matches"]], width=230)

#query by image
image_query = st.text_input(":mag_right: Search for images by image", 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg')
number_of_results = st.slider("Number of results ", 1, 100, 10, key='image')
# image query
if image_query is not None:
    query_vector_image = get_image_embedding(image_query)
    top_k_samples = index.query(
        vector=query_vector_image, top_k=number_of_results, include_values=False
    )
    
st.markdown("<div style='align: center; display: flex'>", unsafe_allow_html=True)
st.image([str(result.id) for result in top_k_samples["matches"]], width=230)
st.markdown("<footer>© 2024 MotMinhTaoCanHet. All rights reserved.</footer>", unsafe_allow_html=True)
