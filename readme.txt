No código, você pode fazer o fit da distribuição de background (equação 8 da figura), do sinal, e o fit simultâneo. Um por vez. Basta comentar a parte que não for usar.




Você pode fazer o fit dos mínimos quadrados ou do log. Basta comentar o que você não quiser.
# residual=(y1-new_hist2_fit[i])**2
  residual=abs(mp.log10(y1)-mp.log10(new_hist2_fit[i]))




Se aparecer um erro "cannot convert mpc to float" (algo assim), é por que o guess não está bom. Em geral guess = [0.1] ou guess = [1.1] funcionam. 



O parâmetro "M" foi calculado previamente, não incluí nesse código como o obtive mas posso inserir depois. Ele mede o melhor tamanho da caixa pra calcular a variância.