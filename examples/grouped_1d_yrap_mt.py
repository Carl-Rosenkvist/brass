import numpy as np
import matplotlib.pyplot as plt

import brass


filename = "/Users/carl/Phd/baryon_stopping/default/out-1/particles_binary.bin"

pdgs = [211, -211, 321, -321, 2212, -2212]

reader = brass.BinaryReader(
    filename,
    ["mass", "p0", "pz", "px", "py", "pdg", "ncoll"],
    skip_elastic=True,
)


# --- y_rap request grouped by pdg ---

y_request = brass.HistogramRequest()
y_request.quantities = ["y_rap"]
y_request.axes = [brass.RegularAxis(100, -5.0, 5.0)]
y_request.group_by = brass.HistogramGroupBy("pdg", pdgs)


# --- mt request grouped by pdg ---

mt_request = brass.HistogramRequest()
mt_request.quantities = ["mt"]
mt_request.axes = [brass.RegularAxis(100, 0.0, 3.0)]
mt_request.group_by = brass.HistogramGroupBy("pdg", pdgs)


# Fill both grouped histograms in one file pass
hists_y, hists_mt = brass.histograms(reader, [y_request, mt_request])

events = reader.particle_blocks_read


# --- dN/dy by PDG ---

plt.figure()

for pdg in pdgs:
    hist = hists_y[pdg]

    counts = hist.values
    edges = np.asarray(hist.edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    dndy = counts / (widths * events)

    plt.plot(centers, dndy, label=str(pdg))

plt.xlabel(r"$y$")
plt.ylabel(r"$1/N_{\mathrm{ev}}\, dN/dy$")
plt.title(r"Rapidity distribution by PDG")
plt.grid(True)
plt.legend()


# --- dN/dmT by PDG ---

plt.figure()

for pdg in pdgs:
    hist = hists_mt[pdg]

    counts = hist.values
    edges = np.asarray(hist.edges[0])
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    dndmt = counts / (widths * events)

    plt.plot(centers, dndmt, label=str(pdg))

plt.xlabel(r"$m_T$")
plt.ylabel(r"$1/N_{\mathrm{ev}}\, dN/dm_T$")
plt.title(r"Transverse mass distribution by PDG")
plt.grid(True)
plt.legend()

plt.show()
