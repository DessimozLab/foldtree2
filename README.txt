Install the dependencies

conda env create --name envname --file=foldtree2.yml

and run foldtree2

python ft2treebuilder.py --model ./models/small/small --mafftmat ./models/small/small_mafft_submat.mtx --submat ./models/small/small_custom_matrix.txt --structures YOURSTRUCTUREFOLDER --outdir RESULTSFOLDER --n_state 35                                          