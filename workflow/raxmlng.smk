
rule mafft_aa:
    conda:
        "foldtree2"
    input:
        fasta="{folder}/sequences.fst"
    output:
        aln="{folder}/alnAA_AA.fasta"
    shell:
        "mafft {input.fasta} > {output.aln}"

rule raxmlng_aa:
    conda:
        "foldtree2"
    input:
        aln="{folder}/alnAA_AA.fasta"
    output:
        tree="{folder}/alnAA_AA.raxml.bestTree"
    params:
        raxmlng="./raxml-ng",
        model="GTR+I+G"
    shell:
        "{params.raxmlng} --msa {input.aln} --model PROTGTR+I+G --prefix {wildcards.folder}/alnAA_AA.raxml --threads 8 --seed 12345 --redo"
