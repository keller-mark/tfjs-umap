import UMAP from './src/umap';

const X = mnist.get(100);

const umap = new UMAP();
umap.fitTransform(X).then((embedding) => {
    console.log(embedding);
});