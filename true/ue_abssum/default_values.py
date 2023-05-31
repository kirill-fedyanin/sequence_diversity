OUTPUT_TYPE = {
    "inference": ["NSP", "MSP", "ENTROPY", "ENTROPY5", "ENTROPY10",
                  "ENTROPY15", "NSP-B", "ENTROPY-S"],
    "inference_unnormalized": ["USP", "ENTROPY-S-U"],
    "mc_ensemble": [
        "ENSP",
        "MNSP",
        "ENSV",
        "BLEUVAR",
        "ROUGE1VAR",
        "ROUGE2VAR",
        "ROUGELVAR",
        "SACREBLEUVAR",
        "BLEUVARDET",
    ],
    "forward": ["EDSLV", "EDSSV", "EDSL", "EDSS", "BALD", "AVGLOSS", "ELOSS"],
    "sampling_ensemble": ["ENSP+sampling", "MNSP+sampling", "ENSV+sampling"],
    "mc_beam": ["ENSP+beam", "MNSP+beam", "ENSV+beam", "BLEUVAR+beam"],
    "sampling_beam": ["ENSP+sampling+beam", "MNSP+sampling+beam", "ENSV+sampling+beam"],
    "aug_ensemble": ["ENSP+aug", "ENSV+aug"],
    "embeddings": ["MD", "NUQ", "DDU", "RDE", "HUE"],
    "embeddings_encoder": [
        "MD-ENCODER",
        "NUQ-ENCODER",
        "DDU-ENCODER",
        "RDE-ENCODER",
        "HUE-ENCODER",
        "EMB-MLP"
    ],
    "embeddings_decoder": [
        "MD-DECODER",
        "NUQ-DECODER",
        "DDU-DECODER",
        "RDE-DECODER",
        "HUE-DECODER",
        "EMB-MLP"
    ],
    "embeddings_diff": [
        "ENC-DIFF"
    ],
    "ensemble": ["EP-SEQ", "EP-TOK", "PE-TOK", "PE-SEQ"],
    "ep_single": ["SEP-SEQ", "SEP-TOK"],
    "no": ["DUE"],
}
REV_OUTPUT_TYPE = {x: key for key, val in OUTPUT_TYPE.items() for x in val}

DEFAULT_SEQ2SEQ_BASE_METRICS = ["rouge", "bleu", "bartscore", "summac"]

DEFAULT_SEQ2SEQ_METRICS = [
    "ROUGE-1",
    "ROUGE-2",
    "ROUGE-L",
    "BLEU",
    "BARTScore-hr",
    "BARTScore-fa",
    "Top 1 Acc",
    "Top all Acc",
    "MRR"
]

DEFAULT_UE_METRICS = ["prr", "rcc"]  # "rcc" coincides with "prr_wo_oracle"

DEFAULT_TOKEN_LEVEL_MEASURES = [
    "total_uncertainty",
    "data_uncertainty",
    "mutual_information",
    "epkl_total_uncertainty",
    "epkl",
    "rmi",
]

TOP_K = [5, 10, 15]

COLORS = [
    "rgb(255, 0, 0)",
    "rgb(255, 255, 0)",
    "rgb(0, 234, 255)",
    "rgb(170, 0, 255)",
    "rgb(255, 127, 0)",
    "rgb(191, 255, 0)",
    "rgb(0, 149, 255)",
    "rgb(255, 0, 170)",
    "rgb(255, 212, 0)",
    "rgb(106, 255, 0)",
    "rgb(0, 64, 255)",
    "rgb(237, 185, 185)",
    "rgb(185, 215, 237)",
    "rgb(231, 233, 185)",
    "rgb(220, 185, 237)",
    "rgb(185, 237, 224)",
    "rgb(143, 35, 35)",
    "rgb(35, 98, 143)",
    "rgb(143, 106, 35)",
    "rgb(107, 35, 143)",
    "rgb(79, 143, 35)",
    "rgb(0, 0, 0)",
    "rgb(115, 115, 115)",
    "rgb(204, 204, 204)",
    "rgb(255, 0, 0)",
    "rgb(255, 255, 0)",
    "rgb(0, 234, 255)",
    "rgb(170, 0, 255)",
    "rgb(255, 127, 0)",
    "rgb(191, 255, 0)",
    "rgb(0, 149, 255)",
    "rgb(255, 0, 170)",
    "rgb(255, 212, 0)",
    "rgb(106, 255, 0)",
    "rgb(0, 64, 255)",
    "rgb(237, 185, 185)",
    "rgb(185, 215, 237)",
    "rgb(231, 233, 185)",
    "rgb(220, 185, 237)",
    "rgb(185, 237, 224)",
    "rgb(143, 35, 35)",
    "rgb(35, 98, 143)",
    "rgb(143, 106, 35)",
    "rgb(107, 35, 143)",
    "rgb(79, 143, 35)",
    "rgb(0, 0, 0)",
    "rgb(115, 115, 115)",
    "rgb(204, 204, 204)",
]

DEFAULT_UE_METHODS = [
    "NSP",
    "NSP-B",
    "MSP",
    "USP",
    "ENTROPY",
    "EP-SEQ",
    "EP-TOK",
    "PE-SEQ",
    "PE-TOK",
    "SEP-SEQ",
    "SEP-TOK",
    # "ENSP"
    # "EDSLV",
    # "EDSSV",
    # "EDSL",
    # "EDSS",
    # "BALD",
    # "AVGLOSS",
    # "ELOSS",
    "BLEUVAR",
    "MD-ENCODER",
    "NUQ-ENCODER",
    "DDU-ENCODER",
    "RDE-ENCODER",
    "MD-DECODER",
    "NUQ-DECODER",
    "DDU-DECODER",
    "RDE-DECODER",
    "ENC-DIFF",
    "EMB-MLP"
]
