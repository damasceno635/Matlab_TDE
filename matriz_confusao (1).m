clear
clc

% carregar dados
data = readtable("iris.data",'FileType','text','ReadVariableNames',false);

X = table2array(data(:,1:4));
labels = table2array(data(:,5));

labels = grp2idx(labels);

% percentual de treino
perc = 0.7;

idx = randperm(length(labels));

Xsh = X(idx,:);
Ysh = labels(idx);

ntrain = round(perc*length(labels));

Xtrain = Xsh(1:ntrain,:);
Ytrain = Ysh(1:ntrain);

Xtest = Xsh(ntrain+1:end,:);
Ytest = Ysh(ntrain+1:end);

classes = unique(Ytrain);

% calcular centroides
centroides = zeros(length(classes),4);

for c = 1:length(classes)
    centroides(c,:) = mean(Xtrain(Ytrain==classes(c),:));
end

% classificação
predicoes = zeros(length(Ytest),1);

for i = 1:length(Ytest)

    dist = zeros(length(classes),1);

    for c = 1:length(classes)
        dist(c) = norm(Xtest(i,:) - centroides(c,:));
    end

    [~,classe_pred] = min(dist);

    predicoes(i) = classe_pred;

end

% matriz de confusão
matriz = confusionmat(Ytest,predicoes);

% gráfico
figure
confusionchart(Ytest,predicoes)
title('Matriz de Confusão - Classificador DMC (Iris)')