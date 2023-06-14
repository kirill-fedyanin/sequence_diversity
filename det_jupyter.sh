#!/bin/bash
det shell start --workspace=Advanced_Computing_Methods --config-file=det_config.yaml \
 -v /home/kirill.fedianin/apps/sequence_uncertainty:/app \
 -v /home/kirill.fedianin/.cache:/app/.cache \
 -- -4 -L 8900:localhost:8900
# -v /home/kirill.fedianin/apps/true:/true \

#jupyter notebook --port=8900 ServerApp.token=36d2cfc0b1921aba751d9776bce474a85e42e7c830f5ed9c