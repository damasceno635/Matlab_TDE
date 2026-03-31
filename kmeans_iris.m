clear; clc;

% --------------------------
% Ler o dataset Iris do CSV
data = readtable('iris.csv');      
X = table2array(data(:,1:4));        
Y_str = data.Var5;                   
Y_real = grp2idx(Y_str);             

% --------------------------
% Configurações do treino
percentuais = 0.1:0.1:0.8;           
num_exec = 20;                        
k = 3;                                
resultados = zeros(length(percentuais), num_exec);  

% --------------------------
% Loop sobre percentuais de treino
for p = 1:length(percentuais)
    perc = percentuais(p);
    
    for rodada = 1:num_exec
        
        % Embaralhar dados
        idx = randperm(size(X,1));
        X_emb = X(idx,:);
        Y_emb = Y_real(idx);
        
        % Definir dados de treino e teste
        n_train = round(perc * size(X,1));
        X_train = X_emb(1:n_train,:);
        Y_train = Y_emb(1:n_train);
        X_test = X_emb(n_train+1:end,:);
        Y_test = Y_emb(n_train+1:end);
        
        % Rodar k-means no treino
        [idx_cluster, C] = kmeans(X_train, k);
        
        % Mapear cada cluster para a classe mais frequente
        mapa = zeros(k,1);
        for c = 1:k
            classe = Y_train(idx_cluster == c);
            if ~isempty(classe)
                mapa(c) = mode(classe);
            end
        end
        
        % Classificar dados de teste
        Y_pred = zeros(size(Y_test));
        for i = 1:length(Y_test)
            dist = sum((C - X_test(i,:)).^2, 2);
            [~, cluster] = min(dist);
            Y_pred(i) = mapa(cluster);
        end
        
        % Calcular acurácia
        acc = sum(Y_pred == Y_test) / length(Y_test);
        resultados(p, rodada) = acc;
    end
end

% --------------------------
% Estatísticas (mínimo, máximo, média e desvio padrão)
for p = 1:length(percentuais)
    dados = resultados(p,:);
    fprintf('Treino %.0f%%:\n', percentuais(p)*100);
    fprintf('  Min: %.2f\n', min(dados));
    fprintf('  Max: %.2f\n', max(dados));
    fprintf('  Media: %.2f\n', mean(dados));
    fprintf('  Desvio: %.2f\n\n', std(dados));
end

% --------------------------
% Salvar resultados em CSV
T = table(percentuais', ...
          min(resultados,[],2), ...
          max(resultados,[],2), ...
          mean(resultados,2), ...
          std(resultados,0,2), ...
          'VariableNames', {'Treino', 'Min', 'Max', 'Media', 'Desvio'});

writetable(T, 'resultados_kmeans.csv');  % cria arquivo CSV com resultados
disp('Arquivo resultados_kmeans.csv salvo');