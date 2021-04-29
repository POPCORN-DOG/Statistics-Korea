install.packages('Ternary')

library('Ternary')

TernaryPlot()

par(mfrow=c(2, 2), mar=rep(0.5, 4))
for (dir in c('up', 'right', 'down', 'le')) {
  TernaryPlot(point=dir, atip='A', btip='B', ctip='C',
              alab='Aness', blab='Bness', clab='Cness')
  TernaryText(list(A=c(10, 1, 1), B=c(1, 10, 1), C=c(1, 1, 10)),
              col=cbPalette8[4], font=2)
}

a <- c(47.35,48.18,46.6,47.46,57.94,58.39,59.28,43.79,36.63)
b <- c(16.32,14.19,20.71,19.47,17.10,16.67,21.16,19.05,18.10)
c <- c(36.34,37.63,32.7,33.06,24.96,24.94,19.56,37.16,45.27)
d <- c('2010','2011','2012','2013','2014','2015','2016','2030','2040')

a_df <- as.data.frame(a)
b_df <- as.data.frame(b)
c_df <- as.data.frame(c)
d_df <- as.data.frame(d)

df <- cbind(a_df,b_df,c_df,d_df)

middle_triangle <- matrix(c(
  30, 40, 30,
  30, 30, 40,
  55, 20, 25
), ncol=3, byrow=TRUE)

TernaryPlot('공공이전', '민간이전','자산재배분',grid.lines=2, grid.lty='dotted',
            grid.minor.lines=1, grid.minor.lty='dotted') +
  TernaryPoints(df[,1:3],col = c('red','blue','green','red','blue','green','red','blue','green')) + TernaryArrows(c(30, 20, 35), c(90, 40, 40), length=0.2, col='darkblue',
                                    lwd =3) +
  legend('bottomright',legend = df[,4])

