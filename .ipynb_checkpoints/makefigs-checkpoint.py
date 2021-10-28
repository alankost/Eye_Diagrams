from ipywidgets import interact, FloatSlider, IntSlider, Button, HBox
import numpy as np
from numpy import fft
from scipy import special
from scipy.interpolate import CubicSpline
from bokeh.io import push_notebook
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, Range1d, LinearAxis, Title, Label, Span
from bokeh.layouts import column, row
from bokeh.models.glyphs import Line
output_notebook()
kB = 1.381e-23 # Boltzmann Constant
T = 300 # Temperature
R = 0.5 # Responsivity
RL = 50 # Load Resistor, Ohms
G = 10 # Post Load Amplifier Gain
bandwidth = 2.5 # Receiver Bandwdith in GHz
sigmaT = np.sqrt(4*kB*T/RL*bandwidth*1e9)*G
Pave = 10e-6; sigma0 = sigmaT; sigma1 = sigmaT # Divide Pave by 1000 to give voltage in mV
Q = (2*Pave*R*RL*G/(sigma0+sigma1+1e-6)) # The Q-Factor that gives BER
BER = 0.5*special.erfc((Pave*2*R*G)/(sigma0+sigma1)/np.sqrt(2)) # Bit Error Rate

v1 = np.random.normal(0, sigma0*RL, 1000)
v2 = np.random.normal(Pave*2*R*RL*G, sigma1*RL, 1000)
v3 = np.random.normal(0, sigma0*RL, 1000)
v = np.concatenate([v1,v2,v3])
t=np.linspace(0,1200,3000) # time in picoseconds
zeros=np.zeros(3000)
Threshold=zeros+Pave*R*RL*G
p = figure(plot_width=400, plot_height=400, background_fill_color="lightgray", title='Signal with Amplitude Noise')
p.title.text_font_style='normal'; p.title.text_font_size='12pt'
p.outline_line_width = 1; p.outline_line_color = "black"; p.min_border_top = 10
l=p.line(t,v,line_color='#2222aa')
lt=p.line(t,Threshold*1e6,line_dash='dashed',line_color='red')
p.x_range = Range1d(0, 1200)
p.y_range = Range1d(-2*1e-3, 10.0*1e-3)
p.xaxis.axis_label="Time (psec)"; p.xaxis.major_label_text_font_size = "12pt"
p.xaxis.axis_label_text_font_style = "normal"; p.xaxis.axis_label_text_font_size = "12pt"
p.yaxis.axis_label="Voltage (mV)"; p.yaxis.major_label_text_font_size = "12pt"
p.yaxis.axis_label_text_font_style = "normal"; p.yaxis.axis_label_text_font_size = "12pt";
label=Label(x=800, y=100, text='Q='+ str(Q), text_color='darkblue', background_fill_color='lightgray')
label1=Label(x=40, y=Pave*R*RL*G*1e6, text='V Threshold', text_color='darkblue', background_fill_color='lightgray')
p.add_layout(label); p.add_layout(label1)

Vvalue=np.linspace(-2*1e-3,10*1e-3,100)
p1 = figure(plot_width=400, plot_height=400, background_fill_color="lightgray", title='Noise Distribution')
p1.title.text_font_style='normal'; p1.title.text_font_size='12pt'
p1.outline_line_width = 1; p1.outline_line_color = "black"
p1.y_range = Range1d(0,1200)
vline = Span(location=0.2,dimension='height', line_color='red', line_dash='dashed')
p1.add_layout(vline)
hist1, edges1 = np.histogram(v1, density=True, bins=100)
hist2, edges2 = np.histogram(v2, density=True, bins=100)
q1 = p1.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
        fill_color="#036564", line_color="#033649")
q2 = p1.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
        fill_color="#036564", line_color="#033649")
ql1 = p1.line(Vvalue, (1/(sigma0*1e5*(2*np.pi)**0.5)*np.exp(-(Vvalue*1e-3-0)**2/(sigma0)**2)), line_width=2, line_color='darkorange')
ql2 = p1.line(Vvalue, (1/(sigma1*1e5*(2*np.pi)**0.5)*np.exp(-(Vvalue*1e-3-Pave*2*R*RL*G)**2/sigma1**2)), line_width=2, line_color='darkorange')
p1.xaxis.axis_label="Voltage (V)"; p1.xaxis.major_label_text_font_size = "12pt"
p1.xaxis.axis_label_text_font_style = "normal"; p1.xaxis.axis_label_text_font_size = "12pt"
p1.yaxis.axis_label="Probabilty Density"; p1.yaxis.major_label_text_font_size = "12pt"
p1.yaxis.axis_label_text_font_style = "normal"; p1.yaxis.axis_label_text_font_size = "12pt"
label2=Label(x=700,  y=0.009, text='BER='+ str(BER),
                                             text_color='darkblue', background_fill_color='lightgray')
label3=Label(x=Pave*R*RL*G, y=1100, text='V Threshold', text_color='darkblue', background_fill_color='lightgray')
p.add_layout(label2); p1.add_layout(label3) 

Pave_slider=FloatSlider(min=0, max=18, step=1, value = 10, description='Pave (\u03BCW)', continuous_update=True)

def replot(Pave):
    Pave = Pave*1e-6 # Give power in Watts
    v1 = np.random.normal(0, sigma0*RL, 1000)
    v2 = np.random.normal(Pave*2*R*RL*G, sigma0*RL, 1000)
    v3 = np.random.normal(0, sigma0*RL, 1000)

    hist1, edges1 = np.histogram(v1, density=True, bins=100)
    hist2, edges2 = np.histogram(v2, density=True, bins=100)
    v = np.concatenate([v1,v2,v3])
    lt.data_source.data['y']=zeros+Pave*R*RL*G
    l.data_source.data['y']=v
    q1.data_source.data['top']=hist1
    q2.data_source.data['top']=hist2
    q1.data_source.data['left']=edges1[:-1]
    q2.data_source.data['left']=edges2[:-1]
    q1.data_source.data['right']=right=edges1[1:]
    q2.data_source.data['right']=right=edges2[1:]
    ql1.data_source.data['y']=1/((sigma0*RL)*(2*np.pi)**0.5)*np.exp(-(Vvalue-0)**2/2/(sigma0*RL)**2)
    ql2.data_source.data['y']=1/((sigma1*RL)*(2*np.pi)**0.5)*np.exp(-(Vvalue-Pave*2*R*RL*G)**2/2/(sigma0*RL)**2)
    vline.location=Pave*R*RL*G
    Q = (2*Pave*R*G)/(sigma0+sigma1); label.text='Q-Factor='+ str(Q)[:3]
    label1.y=Pave*R*RL*G
    BER = 0.5*special.erfc(Q/np.sqrt(2))
    label2.text='BER='+'{:.2e}'.format(BER)
    label3.x=Pave*R*RL*G
    push_notebook(handle=fig_handle)

fig_handle = show(row(p,p1), notebook_handle=True)
interact(replot, Pave=Pave_slider);