{
    "name": "Verifying",
    "description": "Responsible for validating and assessing the novelty of generated peptide sequences by performing homology searches against established biological databases and retrieving relevant annotations.",
    "tools": [
        {
            "tool": "Verify_SwissProt",
            "description": "Conducts a local BLAST sequence alignment against the curated SwissProt database. It retrieves the most homologous sequences, extracts their taxonomic classifications, and pulls cross-referenced database annotations.",
            "input": "folder_path (path to output folder)"
        },
        {
            "tool": "Verify_DBAASP",
            "description": "Conducts a BLAST sequence alignment against the specific DBAASP (Database of Antimicrobial Activity and Structure of Peptides) dataset used to train AMPGAN_v3. It retrieves the closest matching sequence from the training data along with its biological efficacy and physicochemical information to evaluate the exact novelty of the generated peptides.",
            "input": "folder_path (path to output folder)"
        }
    ],
    "context": [
        "This verification module evaluates the output produced by the generation module, translating natural language validation requests from the planning phase into structured database queries.",
        "By comparing generated sequences against both broad (SwissProt) and specialized training (DBAASP) datasets, this process provides critical data on whether a generated peptide is a novel discovery or a replication of a known sequence.",
        "This verification step should be executed as the final stage in the pipeline and requires the FASTA file produced during the structure prediction phase."
    ]
}
