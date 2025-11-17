sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

sudo apt-get install python3-dev

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/hazirliver/boltz-benchmark.git
cd boltz-benchmark
cd runs/
mkdir grid_benchmark **## TODO: REMOVE**
cd grid_benchmark/
export ROOT_DIR="$(pwd)"


git clone https://github.com/dashabalashova/boltz-screen boltz_screen
git clone https://github.com/jwohlwend/boltz.git boltz_vanilla
mkdir boltz_nim
mkdir data/{small_molecules,msa} -p 
mkdir results


cp boltz_screen/examples/screen/* ./data/small_molecules/
cd data/small_molecules
for f in ./*.yaml; do sed -i "/^  - protein:$/,/^  - ligand:/{/^[[:space:]]*sequence:/a\\
      msa: '${ROOT_DIR}/data/msa/fold_cma1_p23946_unpaired_msa_chains_a.a3m'
}" "$f"; done

cd ../../boltz_vanilla/
uv venv -p 3.12
source .venv/bin/activate
uv pip install -e .[cuda]
uv pip install polars "marimo[recommended]"
export BOLTZ2_EXECUTABLE="$(which boltz)"
python run_boltz2_vanilla.py


cd ../boltz_screen/
deactivate
uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip install -e .[cuda]
uv pip install polars "marimo[recommended]"
python run_boltz2_batch.py


cd ../boltz_nim
deactivate
export NGC_API_KEY=<ngc_key>
echo "export NGC_API_KEY=<ngc_key>" >> ~/.bashrc
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p ${LOCAL_NIM_CACHE}
chmod 777 ${LOCAL_NIM_CACHE}
docker run --rm -d --name boltz2 --runtime=nvidia \
    --shm-size=16G \
    -e NGC_API_KEY \
    -v $LOCAL_NIM_CACHE:/opt/nim/.cache \
    -p 8000:8000 \
    nvcr.io/nim/mit/boltz2:1.3.0

curl -X 'GET' \
    'http://localhost:8000/v1/health/ready' \
    -H 'accept: application/json'

uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip install polars "marimo[recommended]"
python run_boltz2_nim.py
docker stop boltz2 2>/dev/null || true

cd ../analysis
deactivate
uv venv -p 3.12 --seed
source .venv/bin/activate
uv pip install polars "marimo[recommended]" seaborn
python analyze_results.py
