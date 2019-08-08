from os import path


class Constants:
    DIRECTORIES = {
        "root": path.abspath(".."),
        "qubbd": "QuBBD",
        "data": "data"
    }

    @staticmethod
    def get_data_dict():
        return {
            "decisions": Constants.DECISIONS,
            "states": Constants.STATES
        }

    FILES = {
        "qubbdv3": "data_QuBBD_v3final.csv"
    }

    DECISIONS = {
        #   decisions 1 columns
        1: {
            "decision": "Decision 1 (Induction Chemo) Y/N",
            "chemo": "Prescribed Chemo (Single/doublet/triplet/quadruplet/none/NOS)",
            "chemo_modification": "Chemo Modification (Y/N)",
            "modification_type": "Modification Type (0= no dose adjustment, 1=dose modified, 2=dose delayed, 3=dose cancelled, 4=dose delayed & modified, 5=regimen modification, 9=unknown)",
            # "chemo_one_hot": ("chemo_single", "chemo_doublet", "chemo_triplet", "chemo_quadruplet", "chemo_none",
            # "chemo_nos")
        },
        #   decision 2 columns
        2: {
            "decision": "Decision 2 (CC / RT alone)",
            "cc_regimen": "CC Regimen(0= none, 1= platinum based, 2= cetuximab based, 3= others, 9=unknown)",
            "cc_modification": "CC modification (Y/N)"
        },
        #   decision 3 columns
        3: {
            "decision": "Decision 3 Neck Dissection (Y/N)"
        }
    }

    STATES = {
        #   state 0 columns
        0: {
            "age": "Age at Diagnosis (Calculated)",
            "pathological_grade": "Pathological Grade",
            "gender": "Gender",
            "race": "Race",
            "laterality": "Tm Laterality (R/L)",
            "subsite": "Tumor subsite (BOT/Tonsil/Soft Palate/Pharyngeal wall/GPS/NOS)",
            "hpv_p16_status": "HPV/P16 status",
            "affected_lymph": "Affected Lymph node cleaned",
            "tcat": "T-category",
            "ncat": "N-category",
            "njcc": "AJCC 8th edition",
            "smoking_status": "Smoking status at Diagnosis (Never/Former/Current)",
            "smoking_packs_per_year": "Smoking status (Packs/Year)",
            "neck_boost": "Neck boost (Y/N)"
        },
        #   state 1 columns
        1: {
            "dlt_present": "DLT (Y/N)",
            "dlt_type": "DLT_Type",
            "dlt_type_columns": {
                "derm": "DLT_Dermatological",
                "neuro": "DLT_Neurological",
                "gastro": "DLT_Gastrointestinal",
                "hemo": "DLT_Hematological",
                "nephro": "DLT_Nephrological",
                "vascular": "DLT_Vascular",
                "pneumonia": "DLT_Infection (Pneumonia)",
                "grade": "DLT_Grade"
            },
            "imaging_state": "No imaging (0=N, 1=Y)",
            "imaging_columns": {
                "cr_prim": "CR Primary",
                "cr_node": "CR Nodal",
                "pr_prim": "PR Primary",
                "pr_node": "PR Nodal",
                "sd_prim": "SD Primary",
                "sd_node": "SD Nodal"
            }
        },
        #   state 2 columns
        2: {
            "dlt": "DLT 2",
            "imaging_columns": {
                "cr_prim": "CR Primary 2",
                "cr_node": "CR Nodal 2",
                "pr_prim": "PR Primary 2",
                "pr_node": "PR Nodal 2",
                "sd_prim": "SD Primary 2",
                "sd_node": "SD Nodal 2"
            }
        }
    }
