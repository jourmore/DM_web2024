import streamlit as st

def main(file_css="./static/style.css"):
    st.set_page_config(page_title="DR-web v.2024", page_icon="./static/mao.ico", layout="wide")

    with open(file_css, encoding='utf-8') as f:
        st.html('<style>{}</style>'.format(f.read())) 

    # st.logo("./static/homelogo.png", link="http://www.nbscal.online/")
    st.logo("./static/logo.png")

def pages_run():
    st.sidebar.title(":blue[**WEB降维和可视化**]")
    pages = {
        "📦WEB降维+可视化+方差" : [
            st.Page("./mlpages/pca_web.py", title="PCA",icon=":material/favorite:"),
            st.Page("./mlpages/lda_web.py", title="LDA",icon=":material/favorite:")
        ],
        # "📦降维可视化" : [
        #     st.Page("./mlpages/tSNE_web.py", title="t-SNE",icon=":material/favorite:"),
        #     st.Page("./mlpages/UMAP_web.py", title="UMAP",icon=":material/favorite:")
        # ],
    }
    pg = st.navigation(pages)
    pg.run()

if __name__ == '__main__':
    main()
    pages_run()
