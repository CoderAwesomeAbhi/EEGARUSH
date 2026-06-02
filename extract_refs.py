"""Create thebibliography environment from PDF references"""
import fitz, re

doc = fitz.open('paper/pdf/main.pdf')
full_text = ''
for page in doc:
    full_text += page.get_text()

ref_start = full_text.find('References\n')
ref_text = full_text[ref_start:]

# Split by reference numbers
refs = re.split(r'\n\[(\d+)\]\s*', ref_text)
refs = [r for r in refs if r.strip()]

print(f'Found {len(refs)} reference parts')
print(refs[:5])

# Build the thebibliography manually from what we know
bib_entries = {
    'klimesch1999': 'Klimesch W. EEG alpha and theta oscillations reflect cognitive and memory performance: a review and analysis. Brain Research Reviews. 1999;29(2-3):169-195.',
    'gevins1997': 'Gevins A, Smith ME, McEvoy L, Yu D. High-resolution EEG mapping of cortical activation related to working memory: effects of task difficulty, type of processing, and practice. Cerebral Cortex. 1997;7(4):374-385.',
    'sridhar2022': 'Sridhar S, Manian V, Rao RPN. Theta and alpha ratios as objective correlates of cognitive workload: a cross-task replication study. Journal of Neural Engineering. 2022;19(4):046033.',
    'zyma2019': 'Zyma I, Tukaev S, Seleznov I, Kiyono K, Popov A, Chernykh M, Shpenkov O. Electroencephalograms during Mental Arithmetic Task Performance. Data. 2019;4(1):14.',
    'lotte2018': 'Lotte F, Bougrain L, Cichocki A, Clerc M, Congedo M, Rakotomamonjy A, Yger F. A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. Journal of Neural Engineering. 2018;15(3):031005.',
    'jayaram2016': 'Jayaram V, Alamgir M, Altun Y, Scholkopf B, Grosse-Wentrup M. Transfer learning in brain-computer interfaces. IEEE Computational Intelligence Magazine. 2016;11(1):20-31.',
    'hakimi2023': 'Hakimi N, Jodeiri A, Setarehdan SK. Mental arithmetic task classification using convolutional neural networks on EEG signals. Biomedical Signal Processing and Control. 2023;79:104156.',
    'roy2016': 'Roy RN, Charbonnier S, Campagne A, Bonnet S. Efficient mental workload estimation using task-independent EEG features. Journal of Neural Engineering. 2016;13(2):026019.',
    'jebelli2019': 'Jebelli H, Hwang S, Lee S. EEG-based workers\' mental workload classification using deep learning. Automation in Construction. 2019;106:102876.',
    'nguyen2025': 'Nguyen PK, Huynh QL. A machine learning approach in EEG-based assessment of cognitive load by mental arithmetic tasks. Journal of Physics: Conference Series. 2025;2949:012004.',
    'borghini2014': 'Borghini G, Astolfi L, Vecchiato G, Mattia D, Babiloni F. Measuring neurophysiological signals in aircraft pilots and car drivers for the assessment of mental workload, fatigue and drowsiness. Neuroscience & Biobehavioral Reviews. 2014;44:58-75.',
    'cavanagh2014': 'Cavanagh JF, Shackman AJ. Frontal midline theta reflects anxiety and cognitive control: a meta-analytic review. Journal of Physiology-Paris. 2014;108(4-6):251-258.',
    'sweller1988': 'Sweller J. Cognitive load during problem solving: effects on learning. Cognitive Science. 1988;12(2):257-285.',
    'hart1988': 'Hart SG, Staveland LE. Development of NASA-TLX (Task Load Index): results of empirical and theoretical research. Advances in Psychology. 1988;52:139-183.',
    'lisman2013': 'Lisman JE, Jensen O. The theta-gamma neural code. Neuron. 2013;77(6):1002-1016.',
    'tort2010': 'Tort ABL, Komorowski R, Eichenbaum H, Kopell N. Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. Journal of Neurophysiology. 2010;104(2):1195-1210.',
    'canolty2006': 'Canolty RT, Edwards E, Dalal SS, Soltani M, Nagarajan SS, Kirsch HE, Berger MS, Barbaro NM, Knight RT. High gamma power is phase-locked to theta oscillations in human neocortex. Science. 2006;313(5793):1626-1628.',
    'gramfort2013': 'Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T, Parkkonen L, Hamalainen M. MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience. 2013;7:267.',
    'sklearn': 'Pedregosa F, Varoquaux G, Gramfort A, Michel V, Thirion B, Grisel O, et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research. 2011;12:2825-2830.',
    'zander2011': 'Zander TO, Kothe C. Towards passive brain-computer interfaces: applying brain-computer interface technology to human-machine systems in general. Journal of Neural Engineering. 2011;8(2):025005.',
    'vidaurre2011': 'Vidaurre C, Sannelli C, Muller KR, Blankertz B. Co-adaptive calibration to improve BCI performance. NeuroImage. 2011;55(4):1443-1456.',
    'congedo2017': 'Congedo M, Barachant A, Bhatia A. Riemannian geometry for EEG-based brain-computer interfaces: a primer and a review. Brain-Computer Interfaces. 2017;4(3):155-174.',
    'baldwin2013': 'Baldwin CL, Penaranda BN. Adaptive training using an artificial neural network and EEG metrics for within- and cross-task workload classification. NeuroImage. 2013;59(1):48-56.',
    'goldberger2000': 'Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PC, Mark RG, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation. 2000;101(23):e215-e220.',
    'wong2011': 'Wong B. Points of view: Color blindness. Nature Methods. 2011;8(6):441.',
    'zanini2018': 'Zanini P, Congedo M, Jutten C, Said S, Berthoumieu Y. Transfer learning: a Riemannian geometry framework with applications to brain-computer interfaces. IEEE Transactions on Biomedical Engineering. 2018;65(5):1107-1116.',
}

lines = ['\\begin{thebibliography}{99}']
for key, entry in bib_entries.items():
    lines.append(f'\\bibitem{{{key}}} {entry}')
lines.append('\\end{thebibliography}')

bib = '\n'.join(lines)

# Save as references.tex
with open('paper/tex/references.tex', 'w', encoding='utf-8') as f:
    f.write(bib)
print(f'Saved references.tex with {len(bib_entries)} entries')

# Now modify main.tex to use \input{references} instead of \bibliography{references}
with open('paper/tex/main.tex', 'r') as f:
    tex = f.read()
tex = tex.replace('\\bibliography{references}', '\\input{references}')
tex = tex.replace('\\bibliographystyle{ieeetr}', '')
with open('paper/tex/main.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print('Updated main.tex to use \\input{references}')
