# Forkscope demo

| Page | Content |
|---|---|
| `repro.html` | Reproduction writeup and measurement limitations |
| `en.html` / `index.html` | English / Chinese experiment walkthrough |
| `lit.html` | Annotated literature |
| `moonshot.html` | Interpretability proposal |

The pages contain their own charts and scripts. Google Fonts requires a network
connection; the experiments and figures render without it.

From the repository root, run `python3 demo/serve_nocache.py` and open
http://127.0.0.1:8901/en.html. The server serves this directory with caching disabled.

`demo/deploy.sh` uploads the HTML files to `sinoark-singapore:/var/www/forkscope/`,
served at http://47.236.93.96/. It requires the `sinoark-singapore` SSH alias.
