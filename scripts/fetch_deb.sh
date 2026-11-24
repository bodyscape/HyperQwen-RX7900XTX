set -e
cd /tmp
U=https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
curl -fsSL -o rocdxg.deb "$U"
ls -l rocdxg.deb; sha256sum rocdxg.deb
echo "--- control:"; dpkg -I rocdxg.deb | sed -n '1,25p'
echo "--- contents:"; dpkg -c rocdxg.deb | awk '{print $1, $3, $6}' | head -30
