# bulk2sc
bulk2sc is the first framework that provides a solid foundation for generating single-cell data from bulk RNA-seq datasets that learns cell type distributions from single cell reference data. bulk2sc consists of three components: scGMVAE, Bulk Encoder, and genVAE, and they are visualized in the following figure:


<div align="center">
    <img src="figures/schematic.png" height="450">
</div>

In its first component, scGMVAE, bulk2sc uses the low-dimensional representation of the single-cell gene expression data that [autoCell (Xu, 2022)](https://pubmed.ncbi.nlm.nih.gov/36814845/) creates using a Gaussian Mixture Model to learn cell type-specific distribution parameters $\mu_k$ and $\sigma_k^2$.


Below, we show four UMAPs that demonstrate the cell type clusters are different stages of bulk2sc: raw input data, reparameterized latent representation from GMM parameters $\mu_k$ and $\sigma_k^2$, reconstructed input data, and generated data.
<div align="center">
<table>
  <tr>
    <td>
      <img src="figures/pbmc3k.png" alt="pbmc3k" width="200"/>
    </td>
    <td>
      <img src="figures/latent.png" alt="latent" width="200" />
    </td>
  </tr>
  <tr>
    <td>
      <img src="figures/reconstructed.png" alt="reconstructed" width="200"/>
    </td>
    <td>
      <img src="figures/generated.png" alt="generated" width="200" />
    </td>
  </tr>
</table>
</div>

## quick start
For a quick start, you can download the PBMC 1K data and pre-trained Bulk Encoder and scDecoder weights [here](https://drive.google.com/drive/folders/1k_jK3tqNvHMoRXBtNtQ8rMrc12fiXkIi?usp=sharing). To run pre-trained model, simply place the unzipped files inside bulk2sc directory and run
```bash
cd bulk2sc
python main.py
```

## custom data
To train with custom data, you will first need to:
0. If cell types are necessary, run ```scType.R``` to them. You will need to modify the script for your specific data and filenames.
1. Modify parameters in utils.py.
2. Modify main.py to adjust filepath.
3. Run ```python main.py```
    
