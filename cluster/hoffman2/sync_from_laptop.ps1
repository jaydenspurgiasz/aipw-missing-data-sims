# Copy this repo to Hoffman2 $HOME for jspurgia.
# Run from PowerShell in the repo root (missing_data_2).

$User = "jspurgia"
$HostName = "hoffman2.oarc.ucla.edu"
$Remote = "${User}@${HostName}:~/research/missing_data_2"

Write-Host "Creating ~/research on Hoffman if needed..."
ssh "${User}@${HostName}" "mkdir -p ~/research"

Write-Host "Uploading repo to $Remote ..."
# Trailing dot copies contents into missing_data_2
scp -r . $Remote

Write-Host "Done. Next on Hoffman (compute node):"
Write-Host "  srun --pty --time=2:00:00 --mem=8G bash"
Write-Host "  bash ~/research/missing_data_2/cluster/hoffman2/bootstrap_once.sh"
