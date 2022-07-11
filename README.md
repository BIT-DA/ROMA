# ROMA
This repository is the official Pytorch implementation for ACMMM2022 paper
"ROMA: Cross-Domain Region Similarity Matching for Unpaired Nighttime Infrared to Daytime Visible Video Translation".[[Arxiv]](https://arxiv.org/abs/2204.12367)

**Examples of Object Detection:**

![detection1](./images/detection1.gif)

![](./images/detection2.gif)

**Examples of Video Fusion**

![fusion](./images/fusion.gif)

More experimental results can be obtained by contacting us.

# Introduction

## Abstract

Infrared cameras are often utilized to enhance the night vision since the visible light cameras exhibit inferior efficacy without sufficient illumination. However, infrared data possesses inadequate color contrast and representation ability attributed to its intrinsic heat-related imaging principle. This makes it arduous to capture and analyze information for human beings, meanwhile hindering its application. Although, the domain gaps between unpaired nighttime infrared and daytime visible videos are even huger than paired ones that captured at the same time, establishing an effective translation mapping will greatly contribute to various fields. In this case, the structural knowledge within nighttime infrared videos and semantic information contained in the translated daytime visible pairs could be utilized simultaneously. To this end, we propose a tailored framework **ROMA** that couples with our introduced cRoss-domain regiOn siMilarity mAtching technique for bridging the huge gaps. To be specific, ROMA could efficiently translate the unpaired nighttime infrared videos into fine-grained daytime visible ones, meanwhile maintain the spatiotemporal consistency via matching the cross-domain region similarity. Furthermore, we design a multiscale region-wise discriminator to distinguish the details from synthesized visible results and real references. Extensive experiments and evaluations for specific applications indicate ROMA outperforms the state-of-the-art methods.  Moreover, we provide a new and challenging dataset encouraging further research for unpaired nighttime infrared and daytime visible video translation, named *InfraredCity*. In particular, it consists of 9 long video clips including City, Highway and Monitor scenarios. All clips could be split into $579,984$ frames in total, which are $20$ times larger than the recently released daytime infrared-to-visible dataset IRVI.

## InfraredCity and InfraredCity-Lite Dataset


<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj" colspan="2">InfraredCity</th>
    <th class="tg-uzvj" colspan="4">Total Frame</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" colspan="2">Nighttime Infrared</td>
    <td class="tg-9wq8" colspan="4">201,856</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="2">Nighttime Visible</td>
    <td class="tg-9wq8" colspan="4">178,698</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="2">Daytime Visible</td>
    <td class="tg-9wq8" colspan="4">199,430</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="6"></td>
  </tr>
  <tr>
    <td class="tg-uzvj" colspan="2">InfraredCity-Lite</td>
    <td class="tg-uzvj">Infrared<br>Train</td>
    <td class="tg-uzvj">Infrared<br>Test</td>
    <td class="tg-uzvj">Visible<br>Train</td>
    <td class="tg-uzvj">Total</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">City</td>
    <td class="tg-9wq8">clearday</td>
    <td class="tg-9wq8">5,538</td>
    <td class="tg-9wq8">1,000</td>
    <td class="tg-9wq8" rowspan="2">5360</td>
    <td class="tg-9wq8" rowspan="2">15,180</td>
  </tr>
  <tr>
    <td class="tg-9wq8">overcast</td>
    <td class="tg-9wq8">2,282</td>
    <td class="tg-9wq8">1,000</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">Highway</td>
    <td class="tg-9wq8">clearday</td>
    <td class="tg-9wq8">4,412</td>
    <td class="tg-9wq8">1,000</td>
    <td class="tg-9wq8" rowspan="2">6,463</td>
    <td class="tg-9wq8" rowspan="2">15,853</td>
  </tr>
  <tr>
    <td class="tg-9wq8">overcast</td>
    <td class="tg-9wq8">2,978</td>
    <td class="tg-9wq8">1,000</td>
  </tr>
  <tr>
    <td class="tg-9wq8" colspan="2">Monitor</td>
    <td class="tg-9wq8">5,612</td>
    <td class="tg-9wq8">500</td>
    <td class="tg-9wq8">4,194</td>
    <td class="tg-9wq8">10,306</td>
  </tr>
</tbody>
</table>

The datasets and their more details are available in [InfiRay](http://openai.raytrontek.com/apply/Infrared_city.html/).
