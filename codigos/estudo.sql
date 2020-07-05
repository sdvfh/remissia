-- ID do paciente: pubcsnum
-- Meses de sobrevivência: srv_time_mon
-- Morte pelo tumor: vsrtsadx
-- Morte por não ser o tumor: odthclass
-- Vivo ou morto: stat_rec

select PUBCSNUM, count(*)
from base_original
where adjm_6value is not null and adjm_6value <> 0
group by PUBCSNUM
order by count(*) desc;

-- Caso com o maior número de tumores já registrado
select *
from base_original
where pubcsnum = 23482728;

-- Caso com adjm_6value não nulo (e sem ser zero)
select *
from base_original
where pubcsnum = 14623436;

select min(YR_BRTH)
from base_original;

select srv_time_mon, count(*)
from base_original
where stat_rec = 1
group by srv_time_mon;


-- Análise do 8 nas colunas de morte por tumor ou não
select vsrtsadx, odthclass, count(*)
from base_original
group by vsrtsadx, odthclass;

select stat_rec, vsrtsadx, count(*)
from base_original
group by stat_rec, vsrtsadx;


select odthclass, count(*)
from base_original
group by odthclass;

select iccc3xwho, count(*)
from base_original
group by iccc3xwho;


select base_original.pubcsnum, count(*)
from base_original
inner join
(
select PUBCSNUM, count(*)
from base_original
where vsrtsadx = 8
and stat_rec = 1
group by PUBCSNUM
) selecionados on selecionados.PUBCSNUM = base_original.PUBCSNUM
group by base_original.pubcsnum
order by count(*) desc

select srv_time_mon
       , *
from base_original
where pubcsnum = 73810012;


-- Primeira tentativa de criação da target (sucesso)
-- Selecionar apenas os pacientes mortos e o óbito causado pelo câncer

select (case
		when srv_time_mon >= 120 then 3
		when srv_time_mon >= 60 then 2
		when srv_time_mon >= 12 then 1 
		when srv_time_mon is null then null else 0 end) as classe_sobrevivencia
	 , count(*)
from base_original

inner join (
	select pubcsnum, vsrtsadx
	from base_original
	where vsrtsadx = 1
) mortos_cancer on mortos_cancer.PUBCSNUM = base_original.PUBCSNUM
group by (case
		when srv_time_mon >= 120 then 3
		when srv_time_mon >= 60 then 2
		when srv_time_mon >= 12 then 1 
		when srv_time_mon is null then null else 0 end)
-- separar a base em treino e teste


select count(*),
    case
        when random() < 0.7 then 'training'
        else 'test'
    end as split
from base_original t
group by case
        when random() < 0.7 then 'training'
        else 'test'
    end
-- separar de acordo com a base (erro)

with ssize as (
    select
        group
    from  to_split_table
    group by group
    having count(*) >= {{ MINIMUM GROUP SIZE }}) -- {{ MINIMUM GROUP SIZE }} = 1 / {{ TEST_THRESHOLD }}
select
    id_aux,
    ts.group,
    case
        when
        cast(row_number() over (partition by ts.group order by rand()) as double) / cast(count() over (partition by ts.group) as double)
        < {{ TEST_THRESHOLD }} then 'test'
        else 'train'
    end as splitting
from  to_split_table ts
join ssize
on ts.group = ssize.group

select srv_time_mon, count(*)
from base_original
where vsrtsadx = 1
group by srv_time_mon
order by srv_time_mon desc

-- Teste para ver se a divisão estratificada foi feita

select seer_teste.tipo_arquivo, count(*)::float / qtd_total
from seer_teste
inner join
(select tipo_arquivo, count(*) qtd_total
from seer
group by tipo_arquivo
)total_original
on total_original.tipo_arquivo = seer_teste.tipo_arquivo
group by seer_teste.tipo_arquivo, qtd_total