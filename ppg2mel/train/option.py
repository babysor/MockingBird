# Default parameters which will be imported by solver
default_hparas = {
    'GRAD_CLIP': 5.0,          # Grad. clip threshold
    'PROGRESS_STEP': 100,      # Std. output refresh freq.
    # Decode steps for objective validation (step = ratio*input_txt_len)
    'DEV_STEP_RATIO': 1.2,
    # Number of examples (alignment/text) to show in tensorboard
    'DEV_N_EXAMPLE': 4,
    'TB_FLUSH_FREQ': 180       # Update frequency of tensorboard (secs)
}
