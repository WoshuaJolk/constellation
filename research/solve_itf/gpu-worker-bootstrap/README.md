# Temporary worker HTTPS bootstrap

These files establish the control connection before workload dependency setup.
They do not create a Droplet or start scientific work. Use a separately bounded
provisioner and deletion guard; powering a Droplet off does not end billing.

Install with cloud-init `write_files`:

- `bootstrap.sh` -> `/root/itf-bootstrap.sh`
- `bootstrap-probe.py` -> `/root/itf-bootstrap-probe.py`
- `stage.py` -> `/root/itf-stage.py`
- The existing authenticated worker agent -> `/root/itf-worker-agent.py`
- A unique random worker credential -> `/root/itf-worker-auth`, mode 0600

Create `itf-worker.service` and `itf-bootstrap-probe.service` with respective
`ExecStart=/usr/bin/python3 /root/itf-worker-agent.py` and
`ExecStart=/usr/bin/python3 /root/itf-bootstrap-probe.py`, `After=network.target`,
`Restart=always`, `WantedBy=multi-user.target`. Run
`bash /root/itf-bootstrap.sh` through cloud-init `runcmd`. The provider PAT must
remain on the controller; never put it in user-data or worker files.

The worker agent must listen on loopback and require its bearer credential on
**every** API route. The sole unauthenticated diagnostic endpoint is
`http://<IP-or-host>/bootstrap-status`: a stage name, timestamp and optional error
line/status. It exposes no credentials, logs or workload data. After TLS starts,
Caddy takes port 80 and returns a fixed `caddy-configured` diagnostic response.
All worker operations use certificate-verified HTTPS on the IP's sslip.io host.

Caddy 2.10.2 is downloaded from its official GitHub release and checked against a
pinned SHA256 before extraction. The preparation receipt also verifies the
release's published SHA512. Caddy is independent of apt and Python workload
packages; package installation cannot prevent initial authenticated access.
The bootstrap checks Caddy configuration before starting it. Failures remain
visible in the minimal diagnostic endpoint and cloud-init logs.

Before any expensive workload, verify unauthorized requests fail, authenticated
status works, upload/download bytes match, a named command finishes, and its
result can be returned. These checks establish transport, not CUDA compilation
or scientific correctness. Collect compact receipts and delete test resources.

## Verified image and transport caveats

Image ID 236925144 reports Ubuntu 22.04.5, Python 3.10.12 and CUDA 13.1. Its
configured apt sources cannot locate `caddy`; do not assume the OS version in
current product documentation describes an older image ID. Workload setup can
install `python3-venv` separately after transport is available.

The managed client proxy returned HTTP 502 for a 10,875,081-byte upload. Sending
the same immutable archive in 512-KiB parts worked. Check each part on receipt,
then verify the SHA256 of their ordered concatenation before extracting. Retry
only identical-byte PUTs on transient transport errors. Never blindly retry a
job-launch POST: first reconcile its unique name with the worker job registry.
