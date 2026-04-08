import foldcomp
# 01. Handling a FCZ file
# Open a fcz file
with open("test/compressed.fcz", "rb") as fcz:
  fcz_binary = fcz.read()

  # Decompress
  (name, pdb) = foldcomp.decompress(fcz_binary) # pdb_out[0]: file name, pdb_out[1]: pdb binary string

  # Save to a pdb file
  with open(name, "w") as pdb_file:
    pdb_file.write(pdb)

  # Get data as dictionary
  data_dict = foldcomp.get_data(fcz_binary) # foldcomp.get_data(pdb) also works
  # Keys: phi, psi, omega, torsion_angles, residues, bond_angles, coordinates
  data_dict["phi"] # phi angles (C-N-CA-C)
  data_dict["psi"] # psi angles (N-CA-C-N)
  data_dict["omega"] # omega angles (CA-C-N-CA)
  data_dict["torsion_angles"] # torsion angles of the backbone as list (phi + psi + omega)
  data_dict["bond_angles"] # bond angles of the backbone as list
  data_dict["residues"] # amino acid residues as string
  data_dict["coordinates"] # coordinates of the backbone as list

# 02. Iterate over a database of FCZ files
# Open a foldcomp database
ids = ["d1asha_", "d1it2a_"]
with foldcomp.open("test/example_db", ids=ids) as db:
  # Iterate through database
  for (name, pdb) in db:
      # save entries as seperate pdb files
      with open(name + ".pdb", "w") as pdb_file:
        pdb_file.write(pdb)