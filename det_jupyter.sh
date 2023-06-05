#!/bin/bash
det shell start --workspace=Advanced_Computing_Methods --config-file=det_config.yaml \
 -v /home/kirill.fedianin/apps/sequence_uncertainty:/app \
 -v /home/kirill.fedianin/.cache:/app/.cache \
 -- -4 -L 8900:localhost:8900


