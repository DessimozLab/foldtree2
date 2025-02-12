Install the dependencies

conda env create --name envname --file=foldtree2.yml

and run foldtree2

python ft2treebuilder.py --model godnodemk5_contactmlp --mafftmat ./models/alpha/mafft_submat.mtx --submat ./models/alpha/raxml_custom_matrix.txt --structures YOURSTRUCTUREFOLDER --outdir RESULTSFOLDER --n_state 38                                             