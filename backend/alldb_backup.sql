--
-- PostgreSQL database cluster dump
--

\restrict uTAhNNyV1VVDP3Uf2b9ogKx8rvdHz4dS1T4XdE3bzJlrtEAfaICgEErhzqziA5h

SET default_transaction_read_only = off;

SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;

--
-- Drop databases (except postgres and template1)
--

DROP DATABASE aistock;
DROP DATABASE mysport;




--
-- Drop roles
--

DROP ROLE postgres;


--
-- Roles
--

CREATE ROLE postgres;
ALTER ROLE postgres WITH SUPERUSER INHERIT CREATEROLE CREATEDB LOGIN REPLICATION BYPASSRLS PASSWORD 'SCRAM-SHA-256$4096:sOHbWsfdclls5j7KP3ac/w==$PM9mVEKJZ7Igt+soy0g5LvqKQFdMKDDgu1+g/MBMY6g=:BIjSrSKhPO0YQoONmTq6k0QbRRpMrG8E9Sl+/fsc5pk=';

--
-- User Configurations
--








\unrestrict uTAhNNyV1VVDP3Uf2b9ogKx8rvdHz4dS1T4XdE3bzJlrtEAfaICgEErhzqziA5h

--
-- Databases
--

--
-- Database "template1" dump
--

--
-- PostgreSQL database dump
--

\restrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L

-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

UPDATE pg_catalog.pg_database SET datistemplate = false WHERE datname = 'template1';
DROP DATABASE template1;
--
-- Name: template1; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE template1 WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE template1 OWNER TO postgres;

\unrestrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L
\connect template1
\restrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE template1; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE template1 IS 'default template for new databases';


--
-- Name: template1; Type: DATABASE PROPERTIES; Schema: -; Owner: postgres
--

ALTER DATABASE template1 IS_TEMPLATE = true;


\unrestrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L
\connect template1
\restrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE template1; Type: ACL; Schema: -; Owner: postgres
--

REVOKE CONNECT,TEMPORARY ON DATABASE template1 FROM PUBLIC;
GRANT CONNECT ON DATABASE template1 TO PUBLIC;


--
-- PostgreSQL database dump complete
--

\unrestrict zBfg6A5imN7zdB58KUYPW99F4Evh9BbS30GlhkqPl8yVG3Luh9ssnGPbSvBJz4L

--
-- Database "aistock" dump
--

--
-- PostgreSQL database dump
--

\restrict QoTezYTxU0wGSlWyyzHUVvNwPwi4fuIysyV5jOHjtg5W4UvvR9NadR6WNPwI7DZ

-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: aistock; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE aistock WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE aistock OWNER TO postgres;

\unrestrict QoTezYTxU0wGSlWyyzHUVvNwPwi4fuIysyV5jOHjtg5W4UvvR9NadR6WNPwI7DZ
\connect aistock
\restrict QoTezYTxU0wGSlWyyzHUVvNwPwi4fuIysyV5jOHjtg5W4UvvR9NadR6WNPwI7DZ

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: orders; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.orders (
    id integer NOT NULL,
    symbol character varying NOT NULL,
    side character varying NOT NULL,
    entry_price double precision NOT NULL,
    exit_price double precision,
    entry_time timestamp without time zone NOT NULL,
    exit_time timestamp without time zone,
    actual_return double precision NOT NULL,
    pnl_amount double precision NOT NULL,
    prediction_id integer,
    status character varying NOT NULL
);


ALTER TABLE public.orders OWNER TO postgres;

--
-- Name: orders_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.orders_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.orders_id_seq OWNER TO postgres;

--
-- Name: orders_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.orders_id_seq OWNED BY public.orders.id;


--
-- Name: predictions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.predictions (
    id integer NOT NULL,
    symbol character varying NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    expected_return double precision NOT NULL,
    confidence_interval_low double precision,
    confidence_interval_high double precision,
    features_snapshot character varying NOT NULL,
    model_version character varying NOT NULL
);


ALTER TABLE public.predictions OWNER TO postgres;

--
-- Name: predictions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.predictions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.predictions_id_seq OWNER TO postgres;

--
-- Name: predictions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.predictions_id_seq OWNED BY public.predictions.id;


--
-- Name: user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public."user" (
    id integer NOT NULL,
    full_name character varying NOT NULL,
    phone character varying NOT NULL,
    role character varying NOT NULL,
    password character varying NOT NULL,
    is_disabled boolean NOT NULL,
    login_retry_count integer NOT NULL,
    openid character varying,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public."user" OWNER TO postgres;

--
-- Name: user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_id_seq OWNER TO postgres;

--
-- Name: user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_id_seq OWNED BY public."user".id;


--
-- Name: orders id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders ALTER COLUMN id SET DEFAULT nextval('public.orders_id_seq'::regclass);


--
-- Name: predictions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions ALTER COLUMN id SET DEFAULT nextval('public.predictions_id_seq'::regclass);


--
-- Name: user id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user" ALTER COLUMN id SET DEFAULT nextval('public.user_id_seq'::regclass);


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
13a907ff29e9
\.


--
-- Data for Name: orders; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.orders (id, symbol, side, entry_price, exit_price, entry_time, exit_time, actual_return, pnl_amount, prediction_id, status) FROM stdin;
1	NVDA	buy	900.5	932	2026-04-15 10:41:26.312734	2026-04-15 17:41:26.312734	0.035	3150	1	closed
2	AAPL	sell	170	175	2026-04-16 10:41:26.312734	2026-04-16 13:41:26.312734	-0.029	-500	2	closed
3	TSLA	buy	180	182.5	2026-04-16 22:41:26.312734	2026-04-17 04:41:26.312734	0.013	250	3	closed
4	BTCUSDT	buy	65000	\N	2026-04-17 08:41:26.312734	\N	0	0	4	open
5	sh512760	buy	0.881	\N	2026-04-22 09:18:22.404878	\N	0	0	5	OPEN
6	sh512760	sell	0.896	0.896	2026-04-22 09:18:41.872029	2026-04-22 09:48:22.381962	0.017026106696935314	213.00000000000017	6	REDUCE
7	sh512760	sell	0.896	0.896	2026-04-22 09:18:41.872029	2026-04-22 10:18:22.378419	0.017026106696935314	106.50000000000009	7	REDUCE
8	sh512760	sell	0.897	0.897	2026-04-22 10:37:24.522507	2026-04-22 10:37:25.912911	0.018161180476731004	57.60000000000005	8	REDUCE
9	sh512760	sell	0.894	0.894	2026-04-22 10:37:24.522507	2026-04-22 11:07:25.919592	0.01475595913734394	23.40000000000002	9	REDUCE
10	sh512760	sell	0.898	0.898	2026-04-22 10:37:24.522507	2026-04-22 11:37:25.901524	0.01929625425652669	15.300000000000013	10	REDUCE
11	sh512760	sell	0.898	0.898	2026-04-22 11:49:10.32744	2026-04-22 13:00:00.292045	0.01929625425652669	8.500000000000007	11	REDUCE
12	sh512760	sell	0.901	0.901	2026-04-22 11:49:10.32744	2026-04-22 13:30:00.36427	0.022701475595913755	6.000000000000005	12	REDUCE
13	sh512760	sell	0.904	0.904	2026-04-22 11:49:10.32744	2026-04-22 14:00:00.351584	0.026106696935300818	4.600000000000004	13	REDUCE
14	sh512760	sell	0.906	0.906	2026-04-22 11:49:10.32744	2026-04-22 14:30:00.309246	0.02837684449489219	2.500000000000002	14	REDUCE
15	sh512760	sell	0.908	0.908	2026-04-22 20:28:30.923279	2026-04-22 20:28:32.547872	0.03064699205448357	0	15	REDUCE
16	sh515150	buy	1.678	\N	2026-04-23 10:00:00.462177	\N	0	0	16	OPEN
17	sz300499	buy	43.16	\N	2026-04-23 10:00:00.462177	\N	0	0	17	OPEN
18	sh603002	buy	13.9	\N	2026-04-23 10:00:00.462177	\N	0	0	18	OPEN
19	sh603002	sell	13.88	13.88	2026-04-23 10:00:50.751345	2026-04-23 10:30:00.419819	-0.0014388489208632786	-15.999999999999659	19	REDUCE
20	sh515150	sell	1.69	1.69	2026-04-23 10:00:03.61391	2026-04-23 10:30:00.419819	0.007151370679380221	50.40000000000005	20	REDUCE
21	sz300499	sell	43.01	43.01	2026-04-23 10:00:30.659251	2026-04-23 10:30:00.419819	-0.0034754402224281417	-29.999999999999716	21	REDUCE
22	sh603002	sell	13.93	13.93	2026-04-23 10:00:50.751345	2026-04-23 11:00:00.451693	0.002158273381294918	11.999999999999744	22	REDUCE
23	sh515150	sell	1.688	1.688	2026-04-23 10:00:03.61391	2026-04-23 11:00:00.451693	0.0059594755661501846	21.000000000000018	23	REDUCE
24	sz300499	sell	42.79	42.79	2026-04-23 10:00:30.659251	2026-04-23 11:00:00.451693	-0.008572752548656104	-36.999999999999744	24	REDUCE
25	sh603002	sell	13.7	13.7	2026-04-23 12:58:10.998345	2026-04-23 12:58:13.497902	-0.01438848920863317	-40.00000000000021	25	REDUCE
26	sh588200	buy	2.704	\N	2026-04-23 12:58:13.497902	\N	0	0	26	OPEN
27	sh515150	sell	1.679	1.679	2026-04-23 12:58:10.998345	2026-04-23 12:58:13.497902	0.0005959475566150846	1.100000000000123	27	REDUCE
28	sz300499	sell	42.29	42.29	2026-04-23 12:58:10.998345	2026-04-23 12:58:13.497902	-0.02015755329008335	-0	28	REDUCE
29	sh603002	sell	13.7	13.7	2026-04-23 12:58:10.998345	2026-04-23 13:00:00.422067	-0.01438848920863317	-20.000000000000107	29	REDUCE
30	sh588200	sell	2.704	2.704	2026-04-23 12:58:55.816805	2026-04-23 13:00:00.422067	0	0	30	REDUCE
31	sh515150	sell	1.679	1.679	2026-04-23 12:58:10.998345	2026-04-23 13:00:00.422067	0.0005959475566150846	0.6000000000000671	31	REDUCE
32	sh512760	buy	0.89	\N	2026-04-23 13:00:00.422067	\N	0	0	32	OPEN
33	sh603002	sell	13.81	13.81	2026-04-23 12:58:10.998345	2026-04-23 13:30:00.38352	-0.0064748201438848815	-0	33	REDUCE
34	sh588200	sell	2.72	2.72	2026-04-23 12:58:55.816805	2026-04-23 13:30:00.38352	0.0059171597633136145	20.80000000000002	34	REDUCE
35	sh515150	sell	1.684	1.684	2026-04-23 12:58:10.998345	2026-04-23 13:30:00.38352	0.0035756853396901106	1.8000000000000016	35	REDUCE
36	sh512760	sell	0.893	0.893	2026-04-23 13:01:10.39879	2026-04-23 13:30:00.38352	0.0033707865168539357	20.700000000000017	36	REDUCE
37	sh588200	sell	2.749	2.749	2026-04-23 12:58:55.816805	2026-04-23 14:00:00.378232	0.0166420118343195	31.49999999999995	37	REDUCE
38	sh515150	sell	1.691	1.691	2026-04-23 12:58:10.998345	2026-04-23 14:00:00.378232	0.007747318235995305	0	38	REDUCE
39	sh512760	sell	0.902	0.902	2026-04-23 13:01:10.39879	2026-04-23 14:00:00.378232	0.013483146067415743	42.000000000000036	39	REDUCE
40	sh588200	sell	2.725	2.725	2026-04-23 12:58:55.816805	2026-04-23 14:30:00.395893	0.0077662721893490775	8.399999999999963	40	REDUCE
41	sh512760	sell	0.894	0.894	2026-04-23 13:01:10.39879	2026-04-23 14:30:00.395893	0.004494382022471914	7.200000000000006	41	REDUCE
42	sh512760	sell	0.897	0.897	2026-04-23 14:52:30.097299	2026-04-23 14:52:32.195297	0.00786516853932585	6.300000000000006	42	REDUCE
43	sh588200	sell	2.736	2.736	2026-04-23 14:52:30.097299	2026-04-23 14:52:32.195297	0.011834319526627229	6.400000000000006	43	REDUCE
44	sz300499	buy	42.38	\N	2026-04-23 14:52:32.195297	\N	0	0	44	OPEN
45	sh512760	sell	0.897	0.897	2026-04-23 14:52:30.097299	2026-04-23 15:00:00.408155	0.00786516853932585	0	45	REDUCE
46	sh588200	sell	2.746	2.746	2026-04-24 09:10:33.614748	2026-04-24 09:30:00.372288	0.015532544378698155	0	46	REDUCE
47	sz300499	sell	41.61	41.61	2026-04-24 09:10:33.614748	2026-04-24 09:30:00.372288	-0.01816894761680045	-154.00000000000063	47	REDUCE
48	sh512760	buy	0.897	\N	2026-04-24 09:30:00.372288	\N	0	0	48	OPEN
49	sz300499	sell	40.07	40.07	2026-04-24 09:10:33.614748	2026-04-24 10:00:00.287829	-0.05450684285040118	-231.00000000000023	49	REDUCE
50	sh512760	sell	0.899	0.899	2026-04-24 09:30:58.925466	2026-04-24 10:00:00.287829	0.002229654403567449	21.40000000000002	50	REDUCE
51	sz002463	buy	103.72	\N	2026-04-24 10:00:00.287829	\N	0	0	51	OPEN
52	sz002463	sell	102.95	102.95	2026-04-24 10:00:55.291255	2026-04-24 10:30:00.309737	-0.007423833397608909	-0	52	REDUCE
53	sz300499	sell	40.1	40.1	2026-04-24 09:10:33.614748	2026-04-24 10:30:00.309737	-0.05379896177442192	-0	53	REDUCE
54	sh512760	sell	0.892	0.892	2026-04-24 09:30:58.925466	2026-04-24 10:30:00.309737	-0.005574136008918622	-27.000000000000025	54	REDUCE
55	sh512760	sell	0.914	0.914	2026-04-24 09:30:58.925466	2026-04-24 11:00:00.373657	0.018952062430323317	45.90000000000004	55	REDUCE
56	sz002463	buy	103.42	\N	2026-04-24 11:00:00.373657	\N	0	0	56	OPEN
57	sz002463	sell	101.34	101.34	2026-04-24 11:00:50.048133	2026-04-24 13:00:00.283683	-0.020112163991490992	-0	57	REDUCE
58	sh512760	sell	0.912	0.912	2026-04-24 09:30:58.925466	2026-04-24 13:00:00.283683	0.016722408026755866	21.000000000000018	58	REDUCE
59	sh512760	sell	0.917	0.917	2026-04-24 09:30:58.925466	2026-04-24 13:30:00.383393	0.02229654403567449	14.000000000000012	59	REDUCE
60	sh512760	sell	0.909	0.909	2026-04-24 09:30:58.925466	2026-04-24 14:00:00.434952	0.013377926421404694	0	60	REDUCE
61	sh603002	buy	13.28	\N	2026-04-29 09:23:34.392684	\N	0	0	61	OPEN
62	sh603002	sell	13.98	13.98	2026-04-29 09:24:43.796307	2026-04-29 09:30:00.386496	0.052710843373494055	1470.0000000000023	62	REDUCE
63	sh603002	sell	14.45	14.45	2026-04-29 09:24:43.796307	2026-04-29 10:00:00.493449	0.08810240963855422	1287	63	REDUCE
64	sh603002	sell	14.44	14.44	2026-04-29 09:24:43.796307	2026-04-29 10:30:00.305379	0.08734939759036146	696.0000000000001	64	REDUCE
65	sh603002	sell	14.51	14.51	2026-04-29 09:24:43.796307	2026-04-29 11:00:00.385055	0.09262048192771088	369.0000000000001	65	REDUCE
66	sh603871	buy	13.31	\N	2026-04-29 11:00:00.385055	\N	0	0	66	OPEN
67	sh603871	sell	13.21	13.21	2026-04-29 11:52:08.487891	2026-04-29 13:00:00.616805	-0.007513148009015751	-69.99999999999974	67	REDUCE
68	sh603002	sell	14.85	14.85	2026-04-29 11:52:08.487891	2026-04-29 13:00:00.616805	0.11822289156626509	314.00000000000006	68	REDUCE
69	sh603871	sell	13.26	13.26	2026-04-29 11:52:08.487891	2026-04-29 13:38:51.677786	-0.003756574004507942	-20.000000000000284	69	REDUCE
70	sh603002	sell	14.5	14.5	2026-04-29 11:52:08.487891	2026-04-29 13:38:51.677786	0.09186746987951812	122.00000000000006	70	REDUCE
71	sh603871	sell	13.26	13.26	2026-04-29 11:52:08.487891	2026-04-29 14:00:00.290449	-0.003756574004507942	-10.000000000000142	71	REDUCE
72	sh603002	sell	14.48	14.48	2026-04-29 11:52:08.487891	2026-04-29 14:00:00.290449	0.09036144578313261	0	72	REDUCE
73	sh603871	sell	13.25	13.25	2026-04-30 09:09:55.513827	2026-04-30 09:30:00.58259	-0.004507888805409504	-6.00000000000005	73	REDUCE
74	sz159908	buy	3.429	\N	2026-04-30 09:30:00.58259	\N	0	0	74	OPEN
75	sh603871	sell	13.11	13.11	2026-04-30 09:09:55.513827	2026-04-30 10:00:00.626128	-0.015026296018031635	-0	75	REDUCE
76	sz159908	sell	3.442	3.442	2026-04-30 09:31:21.697788	2026-04-30 10:00:00.626128	0.0037911927675708212	68.90000000000182	76	REDUCE
77	sh515150	buy	1.672	\N	2026-04-30 10:00:00.626128	\N	0	0	77	OPEN
78	sh515150	sell	1.666	1.666	2026-04-30 10:01:15.497755	2026-04-30 10:30:00.394901	-0.003588516746411487	-68.40000000000006	78	REDUCE
79	sz159908	sell	3.42	3.42	2026-04-30 09:31:21.697788	2026-04-30 10:30:00.394901	-0.0026246719160104687	-24.29999999999972	79	REDUCE
80	sh601899	buy	33.56	\N	2026-04-30 10:30:00.394901	\N	0	0	80	OPEN
81	sh515150	sell	1.664	1.664	2026-04-30 10:01:15.497755	2026-04-30 11:00:00.684948	-0.004784688995215315	-45.60000000000004	81	REDUCE
82	sz159908	sell	3.418	3.418	2026-04-30 09:31:21.697788	2026-04-30 11:00:00.684948	-0.0032079323417905153	-15.399999999999547	82	REDUCE
83	sh601899	sell	33.41	33.41	2026-04-30 10:31:11.848181	2026-04-30 11:00:00.684948	-0.004469606674612803	-120.00000000000455	83	REDUCE
84	sh515150	sell	1.665	1.665	2026-04-30 10:01:15.497755	2026-04-30 13:00:00.767222	-0.004186602870813335	-20.299999999999695	84	REDUCE
85	sz159908	sell	3.43	3.43	2026-04-30 09:31:21.697788	2026-04-30 13:00:00.767222	0.0002916302128901528	0.7000000000002338	85	REDUCE
86	sh601899	sell	33.4	33.4	2026-04-30 10:31:11.848181	2026-04-30 13:00:00.767222	-0.004767580452920253	-64.00000000000148	86	REDUCE
87	sh515150	sell	1.665	1.665	2026-04-30 10:01:15.497755	2026-04-30 13:30:00.566663	-0.004186602870813335	-10.499999999999844	87	REDUCE
88	sz159908	sell	3.43	3.43	2026-04-30 09:31:21.697788	2026-04-30 13:30:00.566663	0.0002916302128901528	0.4000000000001336	88	REDUCE
89	sh601899	sell	33.12	33.12	2026-04-30 10:31:11.848181	2026-04-30 13:30:00.566663	-0.013110846245530536	-88.00000000000097	89	REDUCE
90	sz002463	buy	102.87	\N	2026-04-30 13:30:00.566663	\N	0	0	90	OPEN
91	sh515150	sell	1.668	1.668	2026-04-30 10:01:15.497755	2026-04-30 14:00:00.375832	-0.0023923444976076576	-3.200000000000003	91	REDUCE
92	sz159908	sell	3.426	3.426	2026-04-30 09:31:21.697788	2026-04-30 14:00:00.375832	-0.0008748906386700699	-0.5999999999999339	92	REDUCE
93	sz002463	sell	102.69	102.69	2026-04-30 13:31:14.340161	2026-04-30 14:00:00.375832	-0.0017497812773403986	-36.000000000001364	93	REDUCE
94	sh601899	sell	33.17	33.17	2026-04-30 10:31:11.848181	2026-04-30 14:00:00.375832	-0.011620977353992866	-39.00000000000006	94	REDUCE
95	sh515150	sell	1.667	1.667	2026-04-30 10:01:15.497755	2026-04-30 14:30:00.503633	-0.0029904306220095056	-1.9999999999999574	95	REDUCE
96	sz159908	sell	3.434	3.434	2026-04-30 09:31:21.697788	2026-04-30 14:30:00.503633	0.0014581510644503756	0	96	REDUCE
97	sz002463	sell	102.64	102.64	2026-04-30 13:31:14.340161	2026-04-30 14:30:00.503633	-0.0022358316321571302	-23.000000000000398	97	REDUCE
98	sh601899	sell	33.18	33.18	2026-04-30 10:31:11.848181	2026-04-30 14:30:00.503633	-0.011323003575685416	-0	98	REDUCE
\.


--
-- Data for Name: predictions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.predictions (id, symbol, "timestamp", expected_return, confidence_interval_low, confidence_interval_high, features_snapshot, model_version) FROM stdin;
1	NVDA	2026-04-15 09:41:26.299227	0.035	0.028	0.042	{"rsi": 65.4, "vol_ratio": 1.2, "ma_cross": "golden"}	chronos-bolt
2	AAPL	2026-04-16 09:41:26.299227	-0.015	-0.02	-0.01	{"rsi": 42.1, "vol_ratio": 0.8, "ma_cross": "death"}	chronos-v1
3	TSLA	2026-04-16 21:41:26.299227	0.052	0.04	0.065	{"rsi": 72.8, "vol_ratio": 2.5, "ma_cross": "golden"}	chronos-bolt
4	BTCUSDT	2026-04-17 07:41:26.299227	0.012	0.005	0.018	{"rsi": 55.0, "vol_ratio": 1.5, "ma_cross": "none"}	chronos-v1
5	sh512760	2026-04-22 09:18:22.404878	0.005868351552635431	\N	\N	{"regime": "good", "model_score": 0.7648322582244873, "atr": 0.006753823927222032, "reason": "raw=LONG, raw_score=0.464, final_score=0.464", "confidence": 0.7648322582244873, "gate_mult": 0.607261819144353}	chronos-v2
6	sh512760	2026-04-22 09:48:22.381962	-0.00871272198855877	\N	\N	{"regime": "good", "model_score": 0.5269670486450195, "atr": 0.008062827364065961, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5630333574235767}	chronos-v2
7	sh512760	2026-04-22 10:18:22.378419	-0.009153802879154682	\N	\N	{"regime": "good", "model_score": 0.5489326119422913, "atr": 0.0083972970093307, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5693806933154736}	chronos-v2
8	sh512760	2026-04-22 10:37:25.912911	-0.0081444401293993	\N	\N	{"regime": "good", "model_score": 0.4730445146560669, "atr": 0.007620504035226804, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5576195254607332}	chronos-v2
9	sh512760	2026-04-22 11:07:25.919592	-0.0093085253611207	\N	\N	{"regime": "good", "model_score": 0.46833324432373047, "atr": 0.007845208455485104, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5589992037339521}	chronos-v2
10	sh512760	2026-04-22 11:37:25.901524	-0.008517523296177387	\N	\N	{"regime": "good", "model_score": 0.5250750184059143, "atr": 0.008378541788818437, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5637686522890027}	chronos-v2
11	sh512760	2026-04-22 13:00:00.292045	-0.010820964351296425	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.008378541788818437, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5522536931760617}	chronos-v2
12	sh512760	2026-04-22 13:30:00.36427	-0.012315040454268456	\N	\N	{"regime": "good", "model_score": 0.4055896997451782, "atr": 0.007527995940290697, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5470164216093365}	chronos-v2
13	sh512760	2026-04-22 14:00:00.351584	-0.01464141346514225	\N	\N	{"regime": "good", "model_score": 0.3714424669742584, "atr": 0.006638496283952928, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5465716382827855}	chronos-v2
14	sh512760	2026-04-22 14:30:00.309246	-0.012848451733589172	\N	\N	{"regime": "good", "model_score": 0.38875871896743774, "atr": 0.0064581568980052765, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5493735342805263}	chronos-v2
15	sh512760	2026-04-22 20:28:32.547872	-0.004150728229433298	\N	\N	{"regime": "good", "model_score": 0.3813088834285736, "atr": 0.006131651454773859, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3696661086875977}	chronos-v2
16	sh515150	2026-04-23 10:00:00.462177	-0.0001819398021325469	\N	\N	{"regime": "good", "model_score": 0.5846855044364929, "atr": 0.006231776658460593, "reason": "raw=LONG, raw_score=0.257, final_score=0.257", "confidence": 0.5846855044364929, "gate_mult": 0.43922527058283906}	chronos-v2
17	sz300499	2026-04-23 10:00:00.462177	0.0018651005811989307	\N	\N	{"regime": "good", "model_score": 0.642083466053009, "atr": 0.7415476158248601, "reason": "raw=LONG, raw_score=0.248, final_score=0.248", "confidence": 0.642083466053009, "gate_mult": 0.3858044363283854}	chronos-v2
18	sh603002	2026-04-23 10:00:00.462177	0.017720846459269524	\N	\N	{"regime": "good", "model_score": 0.7485016584396362, "atr": 0.3364003030217504, "reason": "raw=LONG, raw_score=0.424, final_score=0.424", "confidence": 0.7485016584396362, "gate_mult": 0.5661030618976521}	chronos-v2
19	sh603002	2026-04-23 10:30:00.419819	-0.00735191535204649	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.31501359561937936, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3905053026163411}	chronos-v2
20	sh515150	2026-04-23 10:30:00.419819	-0.006854110863059759	\N	\N	{"regime": "good", "model_score": 0.0, "atr": 0.006999113438562237, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.35580385377933516}	chronos-v2
21	sz300499	2026-04-23 10:30:00.419819	-0.013430245220661163	\N	\N	{"regime": "good", "model_score": 0.2954317331314087, "atr": 0.7124523740547347, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.35484784076452525}	chronos-v2
22	sh603002	2026-04-23 11:00:00.451693	0.007226427551358938	\N	\N	{"regime": "good", "model_score": 0.788138747215271, "atr": 0.3026117827591937, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.38271650746754615}	chronos-v2
23	sh515150	2026-04-23 11:00:00.451693	-0.0037136024329811335	\N	\N	{"regime": "good", "model_score": 0.3284704387187958, "atr": 0.006605006958957245, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.43362631856738726}	chronos-v2
24	sz300499	2026-04-23 11:00:00.451693	0.007682973053306341	\N	\N	{"regime": "good", "model_score": 0.7094013690948486, "atr": 0.6751476121821806, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4495817861360824}	chronos-v2
25	sh603002	2026-04-23 12:58:13.497902	0.0018014699453487992	\N	\N	{"regime": "good", "model_score": 0.6593537330627441, "atr": 0.35327844942586045, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3594673231483681}	chronos-v2
26	sh588200	2026-04-23 12:58:13.497902	0.0003268996952101588	\N	\N	{"regime": "good", "model_score": 0.6169482469558716, "atr": 0.029468922953758008, "reason": "raw=LONG, raw_score=0.260, final_score=0.260", "confidence": 0.6169482469558716, "gate_mult": 0.4211114581245193}	chronos-v2
27	sh515150	2026-04-23 12:58:13.497902	0.0014971806667745113	\N	\N	{"regime": "good", "model_score": 0.9260225296020508, "atr": 0.006924339364429599, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4186391067288029}	chronos-v2
28	sz300499	2026-04-23 12:58:13.497902	0.009147335775196552	\N	\N	{"regime": "good", "model_score": 0.7604832649230957, "atr": 0.7538142788488467, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4263326315608076}	chronos-v2
29	sh603002	2026-04-23 13:00:00.422067	-0.0003553654532879591	\N	\N	{"regime": "good", "model_score": 0.6024399399757385, "atr": 0.30884132294668076, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4165682695918804}	chronos-v2
30	sh588200	2026-04-23 13:00:00.422067	-0.0064577581360936165	\N	\N	{"regime": "good", "model_score": 0.2330058515071869, "atr": 0.026176249118662093, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3539245005550523}	chronos-v2
31	sh515150	2026-04-23 13:00:00.422067	-0.0002508436155039817	\N	\N	{"regime": "good", "model_score": 0.5687959790229797, "atr": 0.0061314946734440925, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4136885137682221}	chronos-v2
32	sh512760	2026-04-23 13:00:00.422067	0.0004890929558314383	\N	\N	{"regime": "good", "model_score": 0.6424587965011597, "atr": 0.008265215199790803, "reason": "raw=LONG, raw_score=0.226, final_score=0.226", "confidence": 0.6424587965011597, "gate_mult": 0.35249309304102566}	chronos-v2
33	sh603002	2026-04-23 13:30:00.38352	-0.02209247648715973	\N	\N	{"regime": "neutral", "model_score": 0.36689886450767517, "atr": 0.30748470210934553, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41641703683558223}	chronos-v2
34	sh588200	2026-04-23 13:30:00.38352	-0.0012759777018800378	\N	\N	{"regime": "good", "model_score": 0.5660418272018433, "atr": 0.0265652254187648, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.44676824691500394}	chronos-v2
35	sh515150	2026-04-23 13:30:00.38352	-0.0008248731028288603	\N	\N	{"regime": "good", "model_score": 0.449079304933548, "atr": 0.006931494673444094, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.36747817143504374}	chronos-v2
36	sh512760	2026-04-23 13:30:00.38352	-0.004676962271332741	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.008297144244736673, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4502156725633648}	chronos-v2
37	sh588200	2026-04-23 14:00:00.378232	-0.006768477149307728	\N	\N	{"regime": "good", "model_score": 0.41611966490745544, "atr": 0.027003338221386477, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.42889201816794936}	chronos-v2
38	sh515150	2026-04-23 14:00:00.378232	-0.006368250586092472	\N	\N	{"regime": "good", "model_score": 0.1439143717288971, "atr": 0.006262653607583172, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41712415565079114}	chronos-v2
39	sh512760	2026-04-23 14:00:00.378232	-0.002368293236941099	\N	\N	{"regime": "good", "model_score": 0.4707872271537781, "atr": 0.008440371639245315, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3621949845841384}	chronos-v2
40	sh588200	2026-04-23 14:30:00.395893	-0.004794794600456953	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.02668024232760821, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.48808353483446926}	chronos-v2
41	sh512760	2026-04-23 14:30:00.395893	-0.006956708617508411	\N	\N	{"regime": "good", "model_score": 0.22376564145088196, "atr": 0.008429919904650189, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3637298419199443}	chronos-v2
42	sh512760	2026-04-23 14:52:32.195297	-0.005200488492846489	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.00922991990465019, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4731843890363226}	chronos-v2
43	sh588200	2026-04-23 14:52:32.195297	-0.00815228559076786	\N	\N	{"regime": "good", "model_score": 0.44959408044815063, "atr": 0.02921357566094156, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5140275011123612}	chronos-v2
44	sz300499	2026-04-23 14:52:32.195297	0.00833328627049923	\N	\N	{"regime": "good", "model_score": 0.883402943611145, "atr": 0.6194727883823794, "reason": "raw=LONG, raw_score=0.336, final_score=0.336", "confidence": 0.883402943611145, "gate_mult": 0.3805474609306974}	chronos-v2
45	sh512760	2026-04-23 15:00:00.408155	-0.004666538443416357	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.00922991990465019, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.49794078028037286}	chronos-v2
46	sh588200	2026-04-24 09:30:00.372288	-0.01294098049402237	\N	\N	{"regime": "good", "model_score": 0.2369248867034912, "atr": 0.026770543359793217, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41681912862396614}	chronos-v2
47	sz300499	2026-04-24 09:30:00.372288	-0.025522762909531593	\N	\N	{"regime": "neutral", "model_score": 0.2821250557899475, "atr": 0.656867050556527, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.38936730749253257}	chronos-v2
48	sh512760	2026-04-24 09:30:00.372288	0.003565447637811303	\N	\N	{"regime": "good", "model_score": 0.6795045733451843, "atr": 0.009351623838310561, "reason": "raw=LONG, raw_score=0.359, final_score=0.359", "confidence": 0.6795045733451843, "gate_mult": 0.5286386426420854}	chronos-v2
49	sz300499	2026-04-24 10:00:00.287829	-0.017231397330760956	\N	\N	{"regime": "neutral", "model_score": 0.3004845380783081, "atr": 0.8002169757058131, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.36581097416631586}	chronos-v2
50	sh512760	2026-04-24 10:00:00.287829	-0.008207041770219803	\N	\N	{"regime": "good", "model_score": 0.19323521852493286, "atr": 0.009718698398120465, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3883329557212843}	chronos-v2
51	sz002463	2026-04-24 10:00:00.287829	0.008837798610329628	\N	\N	{"regime": "good", "model_score": 0.874717116355896, "atr": 2.461437126479757, "reason": "raw=LONG, raw_score=0.340, final_score=0.340", "confidence": 0.874717116355896, "gate_mult": 0.3883353304156916}	chronos-v2
52	sz002463	2026-04-24 10:30:00.309737	0.004699173383414745	\N	\N	{"regime": "good", "model_score": 0.8147625923156738, "atr": 2.427912173065347, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.3624242262499331}	chronos-v2
53	sz300499	2026-04-24 10:30:00.309737	0.0038086092099547386	\N	\N	{"regime": "good", "model_score": 0.8055510520935059, "atr": 0.7536115917398559, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.34286801092837893}	chronos-v2
54	sh512760	2026-04-24 10:30:00.309737	-0.0003840223071165383	\N	\N	{"regime": "good", "model_score": 0.5813949704170227, "atr": 0.00936509416725996, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.39739530813551255}	chronos-v2
55	sh512760	2026-04-24 11:00:00.373657	-0.01457294449210167	\N	\N	{"regime": "good", "model_score": 0.3922974765300751, "atr": 0.011783853236580504, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5312080733155932}	chronos-v2
56	sz002463	2026-04-24 11:00:00.373657	0.009847762063145638	\N	\N	{"regime": "good", "model_score": 0.7683936953544617, "atr": 1.9808762647378975, "reason": "raw=LONG, raw_score=0.333, final_score=0.333", "confidence": 0.7683936953544617, "gate_mult": 0.4336176148502493}	chronos-v2
57	sz002463	2026-04-24 13:00:00.283683	0.014068380929529667	\N	\N	{"regime": "good", "model_score": 0.9100510478019714, "atr": 2.3315429314045635, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41738804122966877}	chronos-v2
58	sh512760	2026-04-24 13:00:00.283683	-0.011855507269501686	\N	\N	{"regime": "good", "model_score": 0.3991706371307373, "atr": 0.011789752567877627, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5354614176267445}	chronos-v2
59	sh512760	2026-04-24 13:30:00.383393	-0.011683127842843533	\N	\N	{"regime": "good", "model_score": 0.40741780400276184, "atr": 0.011065976432779576, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5293092459249353}	chronos-v2
60	sh512760	2026-04-24 14:00:00.434952	-0.004081344231963158	\N	\N	{"regime": "good", "model_score": 0.31755656003952026, "atr": 0.010589888503490992, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.35449690577929877}	chronos-v2
61	sh603002	2026-04-29 09:23:34.392684	0.10386209934949875	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.3234889271113377, "reason": "raw=LONG, raw_score=0.556, final_score=0.556", "confidence": 1.0, "gate_mult": 0.5557437667951736}	chronos-v2
62	sh603002	2026-04-29 09:30:00.386496	0.03877861052751541	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.38302373721822436, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4386637692774549}	chronos-v2
63	sh603002	2026-04-29 10:00:00.493449	0.004839247092604637	\N	\N	{"regime": "good", "model_score": 0.6661190986633301, "atr": 0.3625316829431982, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.49392184537645994}	chronos-v2
64	sh603002	2026-04-29 10:30:00.305379	0.003155551850795746	\N	\N	{"regime": "good", "model_score": 0.6585829854011536, "atr": 0.35223856941472037, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4119430822925546}	chronos-v2
65	sh603002	2026-04-29 11:00:00.385055	0.01191568374633789	\N	\N	{"regime": "good", "model_score": 0.7948951721191406, "atr": 0.3254512045685596, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4585572084229154}	chronos-v2
66	sh603871	2026-04-29 11:00:00.385055	-0.006552179343998432	\N	\N	{"regime": "good", "model_score": 0.4509648382663727, "atr": 0.16882586586171713, "reason": "raw=LONG, raw_score=0.174, final_score=0.174", "confidence": 0.4509648382663727, "gate_mult": 0.38513831483702166}	chronos-v2
67	sh603871	2026-04-29 13:00:00.616805	-0.001142142340540886	\N	\N	{"regime": "good", "model_score": 0.5971940755844116, "atr": 0.1576110919386952, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4040209101390915}	chronos-v2
68	sh603002	2026-04-29 13:00:00.616805	-0.011247730813920498	\N	\N	{"regime": "neutral", "model_score": 0.5, "atr": 0.35494659972682774, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.42257555114351325}	chronos-v2
69	sh603871	2026-04-29 13:38:51.677786	-0.002717820694670081	\N	\N	{"regime": "good", "model_score": 0.5444723963737488, "atr": 0.15363546759601238, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.38522979991701806}	chronos-v2
70	sh603002	2026-04-29 13:38:51.677786	0.00453566201031208	\N	\N	{"regime": "good", "model_score": 0.7030719518661499, "atr": 0.356509275283497, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.411553875824984}	chronos-v2
71	sh603871	2026-04-29 14:00:00.290449	-0.0021807209122925997	\N	\N	{"regime": "good", "model_score": 0.5470011234283447, "atr": 0.15363546759601238, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.37427497486260286}	chronos-v2
72	sh603002	2026-04-29 14:00:00.290449	0.00869583711028099	\N	\N	{"regime": "good", "model_score": 0.8719595670700073, "atr": 0.31413026051878157, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4071309086022957}	chronos-v2
73	sh603871	2026-04-30 09:30:00.58259	-0.010427813977003098	\N	\N	{"regime": "neutral", "model_score": 0.4046163260936737, "atr": 0.14282211681975618, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41443233576318633}	Kronos
74	sz159908	2026-04-30 09:30:00.58259	0.00484090531244874	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.0200039616738255, "reason": "raw=LONG, raw_score=0.349, final_score=0.349", "confidence": 1.0, "gate_mult": 0.34892369249179733}	Kronos
75	sh603871	2026-04-30 10:00:00.626128	-0.0017639847937971354	\N	\N	{"regime": "good", "model_score": 0.6100518703460693, "atr": 0.16548878348642282, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5160850600011604}	Kronos
76	sz159908	2026-04-30 10:00:00.626128	0.0013782767346128821	\N	\N	{"regime": "good", "model_score": 0.6694110631942749, "atr": 0.023070628340492186, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4885004751389239}	Kronos
77	sh515150	2026-04-30 10:00:00.626128	0.0012718461221083999	\N	\N	{"regime": "good", "model_score": 0.9737561941146851, "atr": 0.006010680889615809, "reason": "raw=LONG, raw_score=0.369, final_score=0.369", "confidence": 0.9737561941146851, "gate_mult": 0.37867703015579113}	Kronos
78	sh515150	2026-04-30 10:30:00.394901	0.0034246554132550955	\N	\N	{"regime": "good", "model_score": 0.8421919941902161, "atr": 0.006142589298071155, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5805913069252807}	Kronos
79	sz159908	2026-04-30 10:30:00.394901	-0.0007441029883921146	\N	\N	{"regime": "good", "model_score": 0.5877646803855896, "atr": 0.023857052500896217, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4184795462116829}	Kronos
80	sh601899	2026-04-30 10:30:00.394901	0.011642669327557087	\N	\N	{"regime": "good", "model_score": 0.8595787286758423, "atr": 0.34943204171382913, "reason": "raw=LONG, raw_score=0.533, final_score=0.533", "confidence": 0.8595787286758423, "gate_mult": 0.619969077116489}	Kronos
81	sh515150	2026-04-30 11:00:00.684948	0.0035897328052669764	\N	\N	{"regime": "good", "model_score": 0.8136036396026611, "atr": 0.005590245222929771, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6321396234271404}	Kronos
82	sz159908	2026-04-30 11:00:00.684948	-0.010384754277765751	\N	\N	{"regime": "good", "model_score": 0.29495590925216675, "atr": 0.021162182915819294, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4414466743648654}	Kronos
83	sh601899	2026-04-30 11:00:00.684948	0.019493913277983665	\N	\N	{"regime": "good", "model_score": 0.9963985681533813, "atr": 0.29108453924011285, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6446603736991222}	Kronos
84	sh515150	2026-04-30 13:00:00.767222	0.002960328245535493	\N	\N	{"regime": "good", "model_score": 0.7738535404205322, "atr": 0.005244881656785778, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6278057435309986}	Kronos
85	sz159908	2026-04-30 13:00:00.767222	-0.006839422509074211	\N	\N	{"regime": "good", "model_score": 0.37106090784072876, "atr": 0.021592487116213678, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.43206811628162495}	Kronos
86	sh601899	2026-04-30 13:00:00.767222	0.015704354271292686	\N	\N	{"regime": "good", "model_score": 0.9083587527275085, "atr": 0.2744954741188342, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6438534874040317}	Kronos
87	sh515150	2026-04-30 13:30:00.566663	0.0023015339393168688	\N	\N	{"regime": "good", "model_score": 0.7507975101470947, "atr": 0.004812228245911229, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6207985385814249}	Kronos
88	sz159908	2026-04-30 13:30:00.566663	-0.003439816879108548	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 0.021992487116213634, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.41899410426337536}	Kronos
89	sh601899	2026-04-30 13:30:00.566663	0.025871573016047478	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.2686516207693564, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6477810122850332}	Kronos
90	sz002463	2026-04-30 13:30:00.566663	0.003039950504899025	\N	\N	{"regime": "good", "model_score": 0.6843234300613403, "atr": 1.4120982592877531, "reason": "raw=LONG, raw_score=0.324, final_score=0.324", "confidence": 0.6843234300613403, "gate_mult": 0.4738872250839702}	Kronos
91	sh515150	2026-04-30 14:00:00.375832	0.0001287788909394294	\N	\N	{"regime": "good", "model_score": 0.6482241153717041, "atr": 0.004703930758255921, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5614343292260886}	Kronos
92	sz159908	2026-04-30 14:00:00.375832	-0.0011312842834740877	\N	\N	{"regime": "good", "model_score": 0.5395715832710266, "atr": 0.0202643142282488, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.37427294175060066}	Kronos
93	sz002463	2026-04-30 14:00:00.375832	-0.0149415023624897	\N	\N	{"regime": "good", "model_score": 0.4166043698787689, "atr": 1.3098604677445675, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5078229563588864}	Kronos
94	sh601899	2026-04-30 14:00:00.375832	0.018227607011795044	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.24589807071565023, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6446372442536923}	Kronos
95	sh515150	2026-04-30 14:30:00.503633	0.0017759393667802215	\N	\N	{"regime": "good", "model_score": 0.7438764572143555, "atr": 0.00447673979638821, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.5634788693053342}	Kronos
96	sz159908	2026-04-30 14:30:00.503633	-0.0029695748817175627	\N	\N	{"regime": "good", "model_score": 0.38809680938720703, "atr": 0.016440731969683174, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.35502877546504236}	Kronos
97	sz002463	2026-04-30 14:30:00.503633	-0.007567060180008411	\N	\N	{"regime": "good", "model_score": 0.5, "atr": 1.1892572712030265, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.4868210665265575}	Kronos
98	sh601899	2026-04-30 14:30:00.503633	0.021807145327329636	\N	\N	{"regime": "good", "model_score": 1.0, "atr": 0.22706722363817283, "reason": "equity_slope_break-slope \\u5d29\\u574f\\u4fdd\\u62a4", "confidence": 1.0, "gate_mult": 0.6468253265615429}	Kronos
\.


--
-- Data for Name: user; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public."user" (id, full_name, phone, role, password, is_disabled, login_retry_count, openid, created_at) FROM stdin;
\.


--
-- Name: orders_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.orders_id_seq', 98, true);


--
-- Name: predictions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.predictions_id_seq', 98, true);


--
-- Name: user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_id_seq', 1, false);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: orders orders_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_pkey PRIMARY KEY (id);


--
-- Name: predictions predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (id);


--
-- Name: user user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT user_pkey PRIMARY KEY (id);


--
-- Name: ix_orders_entry_time; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_orders_entry_time ON public.orders USING btree (entry_time);


--
-- Name: ix_orders_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_orders_symbol ON public.orders USING btree (symbol);


--
-- Name: ix_predictions_symbol; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_predictions_symbol ON public.predictions USING btree (symbol);


--
-- Name: ix_predictions_timestamp; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_predictions_timestamp ON public.predictions USING btree ("timestamp");


--
-- Name: ix_user_openid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_openid ON public."user" USING btree (openid);


--
-- Name: ix_user_phone; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_user_phone ON public."user" USING btree (phone);


--
-- Name: orders orders_prediction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT orders_prediction_id_fkey FOREIGN KEY (prediction_id) REFERENCES public.predictions(id);


--
-- PostgreSQL database dump complete
--

\unrestrict QoTezYTxU0wGSlWyyzHUVvNwPwi4fuIysyV5jOHjtg5W4UvvR9NadR6WNPwI7DZ

--
-- Database "mysport" dump
--

--
-- PostgreSQL database dump
--

\restrict LKO09nzkYpC3n8yYhYCbY0Ub2aR8ZJj4eQ3BAIp6W5HpHC1iE7mK6EKFcQbQtdX

-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: mysport; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE mysport WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE mysport OWNER TO postgres;

\unrestrict LKO09nzkYpC3n8yYhYCbY0Ub2aR8ZJj4eQ3BAIp6W5HpHC1iE7mK6EKFcQbQtdX
\connect mysport
\restrict LKO09nzkYpC3n8yYhYCbY0Ub2aR8ZJj4eQ3BAIp6W5HpHC1iE7mK6EKFcQbQtdX

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: scopetype; Type: TYPE; Schema: public; Owner: postgres
--

CREATE TYPE public.scopetype AS ENUM (
    'DISTRICT',
    'STREET',
    'COMMUNITY',
    'SITE',
    'VENUE'
);


ALTER TYPE public.scopetype OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: community; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.community (
    id integer NOT NULL,
    name character varying NOT NULL,
    contact_name character varying,
    contact_phone character varying,
    address character varying,
    street_id integer NOT NULL
);


ALTER TABLE public.community OWNER TO postgres;

--
-- Name: community_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.community_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.community_id_seq OWNER TO postgres;

--
-- Name: community_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.community_id_seq OWNED BY public.community.id;


--
-- Name: dictionary; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dictionary (
    id integer NOT NULL,
    dict_code character varying NOT NULL,
    item_key character varying NOT NULL,
    item_value character varying NOT NULL,
    sort_order integer NOT NULL,
    is_system boolean NOT NULL
);


ALTER TABLE public.dictionary OWNER TO postgres;

--
-- Name: dictionary_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.dictionary_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.dictionary_id_seq OWNER TO postgres;

--
-- Name: dictionary_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.dictionary_id_seq OWNED BY public.dictionary.id;


--
-- Name: equipment; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.equipment (
    name character varying NOT NULL,
    category character varying NOT NULL,
    supplier character varying,
    description character varying,
    id integer NOT NULL,
    created_at timestamp without time zone NOT NULL,
    photo character varying
);


ALTER TABLE public.equipment OWNER TO postgres;

--
-- Name: equipment_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.equipment_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.equipment_id_seq OWNER TO postgres;

--
-- Name: equipment_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.equipment_id_seq OWNED BY public.equipment.id;


--
-- Name: inspection; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.inspection (
    id integer NOT NULL,
    venue_id integer NOT NULL,
    inspector_id integer NOT NULL,
    current_location character varying NOT NULL,
    status integer NOT NULL,
    description character varying,
    created_at timestamp without time zone NOT NULL,
    inspection_time timestamp without time zone NOT NULL,
    photo_url json,
    community_id integer NOT NULL
);


ALTER TABLE public.inspection OWNER TO postgres;

--
-- Name: inspection_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.inspection_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.inspection_id_seq OWNER TO postgres;

--
-- Name: inspection_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.inspection_id_seq OWNED BY public.inspection.id;


--
-- Name: maintenance; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.maintenance (
    id integer NOT NULL,
    inspection_id integer,
    venue_id integer NOT NULL,
    equipment_id integer,
    equipment_name character varying,
    maintainer_id integer,
    issue_type character varying,
    status integer NOT NULL,
    action_taken character varying,
    before_photo json,
    completion_photo json,
    create_user_id integer,
    created_at timestamp without time zone NOT NULL,
    reported_at timestamp without time zone NOT NULL,
    finished_at timestamp without time zone,
    is_urgent boolean,
    completion_remark character varying,
    update_at timestamp without time zone
);


ALTER TABLE public.maintenance OWNER TO postgres;

--
-- Name: maintenance_history; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.maintenance_history (
    id integer NOT NULL,
    maintaine_id integer,
    maintainer_id integer,
    maintainer_name character varying,
    action integer,
    create_user_id integer NOT NULL,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public.maintenance_history OWNER TO postgres;

--
-- Name: maintenance_history_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.maintenance_history_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.maintenance_history_id_seq OWNER TO postgres;

--
-- Name: maintenance_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.maintenance_history_id_seq OWNED BY public.maintenance_history.id;


--
-- Name: maintenance_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.maintenance_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.maintenance_id_seq OWNER TO postgres;

--
-- Name: maintenance_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.maintenance_id_seq OWNED BY public.maintenance.id;


--
-- Name: street; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.street (
    id integer NOT NULL,
    name character varying NOT NULL,
    contact_name character varying,
    contact_phone character varying,
    region character varying NOT NULL
);


ALTER TABLE public.street OWNER TO postgres;

--
-- Name: street_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.street_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.street_id_seq OWNER TO postgres;

--
-- Name: street_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.street_id_seq OWNED BY public.street.id;


--
-- Name: user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public."user" (
    id integer NOT NULL,
    full_name character varying NOT NULL,
    phone character varying NOT NULL,
    role character varying NOT NULL,
    password character varying NOT NULL,
    is_disabled boolean NOT NULL,
    login_retry_count integer NOT NULL,
    openid character varying,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public."user" OWNER TO postgres;

--
-- Name: user_data_permissions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.user_data_permissions (
    id integer NOT NULL,
    user_id integer NOT NULL,
    scope_type public.scopetype NOT NULL,
    scope_id character varying NOT NULL,
    scope_name character varying
);


ALTER TABLE public.user_data_permissions OWNER TO postgres;

--
-- Name: user_data_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_data_permissions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_data_permissions_id_seq OWNER TO postgres;

--
-- Name: user_data_permissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_data_permissions_id_seq OWNED BY public.user_data_permissions.id;


--
-- Name: user_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_id_seq OWNER TO postgres;

--
-- Name: user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_id_seq OWNED BY public."user".id;


--
-- Name: venue; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.venue (
    id integer NOT NULL,
    management_unit character varying,
    owner_unit character varying,
    floor_type character varying,
    address character varying,
    contact_name character varying,
    contact_phone character varying,
    warranty_period character varying,
    area_size character varying,
    venue_type character varying,
    equipment_type character varying,
    equipment_brand character varying,
    venue_count integer,
    equipment_count integer,
    install_date character varying,
    community_id integer NOT NULL,
    name character varying NOT NULL,
    photo_url json,
    district_id character varying,
    street_id integer,
    equipment_detail character varying,
    damage_description character varying,
    damage_count integer DEFAULT 0
);


ALTER TABLE public.venue OWNER TO postgres;

--
-- Name: venue_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.venue_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.venue_id_seq OWNER TO postgres;

--
-- Name: venue_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.venue_id_seq OWNED BY public.venue.id;


--
-- Name: community id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.community ALTER COLUMN id SET DEFAULT nextval('public.community_id_seq'::regclass);


--
-- Name: dictionary id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dictionary ALTER COLUMN id SET DEFAULT nextval('public.dictionary_id_seq'::regclass);


--
-- Name: equipment id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.equipment ALTER COLUMN id SET DEFAULT nextval('public.equipment_id_seq'::regclass);


--
-- Name: inspection id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.inspection ALTER COLUMN id SET DEFAULT nextval('public.inspection_id_seq'::regclass);


--
-- Name: maintenance id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance ALTER COLUMN id SET DEFAULT nextval('public.maintenance_id_seq'::regclass);


--
-- Name: maintenance_history id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance_history ALTER COLUMN id SET DEFAULT nextval('public.maintenance_history_id_seq'::regclass);


--
-- Name: street id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.street ALTER COLUMN id SET DEFAULT nextval('public.street_id_seq'::regclass);


--
-- Name: user id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user" ALTER COLUMN id SET DEFAULT nextval('public.user_id_seq'::regclass);


--
-- Name: user_data_permissions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_data_permissions ALTER COLUMN id SET DEFAULT nextval('public.user_data_permissions_id_seq'::regclass);


--
-- Name: venue id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.venue ALTER COLUMN id SET DEFAULT nextval('public.venue_id_seq'::regclass);


--
-- Data for Name: alembic_version; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.alembic_version (version_num) FROM stdin;
ff5ac5d64e20
\.


--
-- Data for Name: community; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.community (id, name, contact_name, contact_phone, address, street_id) FROM stdin;
1	油松社区	钟家燕	13420955222	\N	1
2	清湖社区	李琳	15813801799	\N	1
3	富康社区	罗云青	13809899348	\N	1
4	三联社区	李景龙	13424431439	\N	1
5	景龙社区	黄小姐	13590283306	\N	1
6	龙园社区	-	15818546857	\N	1
7	华联社区	杨先生	13510535992	\N	1
8	清华社区	麦石好	13723752495	\N	1
9	玉翠社区	陈欢	13824379178	\N	1
10	景新社区	周蓉蓉	19210053296	\N	1
11	松和社区	李姐	13651467206	\N	1
12	和联社区	霞姐	13714001670	\N	1
13	君子布社区	何欣庭	15014044094	\N	2
14	桂花社区	陈巧茹	15999586300	\N	2
15	章阁社区	黄国清	15725580009	\N	2
16	库坑社区	杨金凤	13600431910	\N	2
17	桂香社区	罗媛媛	21031607	\N	2
18	桂香社区	杨金凤	13600431910	\N	2
19	牛湖社区	汪昆	13689503663	\N	2
20	大水田社区	叶小姐	13530466708	\N	2
21	大富社区	陈先生	13392194602	\N	2
22	新澜社区	陈业琪	13751110305	\N	2
23	广培社区	-	21059466	\N	2
24	新牛社区	詹先生	13632731334	\N	3
25	北站社区	俊逸	18278657087	\N	3
26	大岭社区	段小姐	18569665058	\N	3
27	民新社区	张生	13714678598	\N	3
28	民乐社区	潘嘉城	13148766698	\N	3
29	民治社区	舒畅	18998916326	\N	3
30	白石龙社区	曾俏玲	13715227547	\N	3
31	上芬社区	李女士	13691913904	\N	3
32	龙塘社区	段佳佳	29787502	\N	3
33	民泰社区	徐观平	13534171844	\N	3
34	民强社区	苏韦宇	13510260098	\N	3
35	民康社区	沈军	13530814829	\N	3
36	红山社区	林先生	19129355107	\N	3
37	樟坑社区	-	15807643792	\N	3
38	丹湖社区	张伟龙	13714808911	\N	4
39	章阁社区	杨庆发	13824323100	\N	4
40	大水坑社区	黄先生	15013696660	\N	4
41	茜坑社区	陈挺	13751091955	\N	4
42	新和社区	余志明	19925184310	\N	4
43	四和社区	吴先生	18218178365	\N	4
44	兴富社区	刘小姐	13115268557	\N	4
45	润城社区	沈娇露	18123765610	\N	5
46	观城社区	胡欣秀	18822894603	\N	5
47	鹭湖社区	-	23737381	\N	5
48	樟坑径社区	陈思敏	13641446859	\N	5
49	新田社区	郑小姐	21089641	\N	5
50	樟溪社区	卓小姐	13244853610	\N	5
51	大和社区	周牡	18825232251	\N	5
52	新源社区	余先生	18318871794	\N	5
53	同胜社区	-	18816823069	\N	6
54	大浪社区	-	21013182	\N	6
55	浪口社区	-	13510681098	\N	6
56	新石社区	何小姐	13691821215	\N	6
57	高峰社区	-	13728671543	\N	6
58	龙胜社区	-	13713970552	\N	6
59	龙平社区	/	-	\N	6
60	陶元社区	/	-	\N	6
\.


--
-- Data for Name: dictionary; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dictionary (id, dict_code, item_key, item_value, sort_order, is_system) FROM stdin;
1	REGION	NS	南山区	1	t
2	REGION	FT	福田区	2	t
3	REGION	BA	宝安区	3	t
4	REGION	LH	龙华区	4	t
5	MAINTAIN_TYPE	ROUTINE	日常保养	1	f
6	MAINTAIN_TYPE	REPAIR	故障维修	2	f
7	MAINTAIN_TYPE	REPLACE	零件更换	3	f
8	MAINTAIN_TYPE	EMERGENCY	紧急抢修	4	f
9	VENUE_TYPE	BASKETBALL	篮球场	1	f
10	VENUE_TYPE	FITNESS_PATH	健身路径	2	f
11	VENUE_TYPE	PINGPONG	乒乓球场	3	f
12	VENUE_TYPE	FOOTBALL	足球场	4	f
13	EQUIPMENT_TYPE	STRENGTH	力量训练器材	1	f
14	EQUIPMENT_TYPE	AEROBIC	有氧训练器材	2	f
15	EQUIPMENT_TYPE	PATH_SINGLE_BAR	单双杠/路径	3	f
16	EQUIPMENT_TYPE	REHABILITATION	康复/拉伸器材	4	f
17	EQUIPMENT_TYPE	CHILDREN_PLAY	儿童游乐设施	5	f
18	EQUIPMENT_TYPE	OUTDOOR_FITNESS	室外健身设施	6	f
19	EQUIPMENT_TYPE	ACCESSORY	辅助/小件器材	7	f
20	ISSUE_TYPE	GROUND_ISSUE	地面破损	1	f
21	ISSUE_TYPE	EQUIPMENT_ISSUE	器材损坏	2	f
22	ISSUE_TYPE	SAFETY_ISSUE	安全隐患	3	f
23	ISSUE_TYPE	CLEANLINESS_ISSUE	卫生问题	4	f
24	ISSUE_TYPE	OTHER_ISSUE	其他问题	5	f
\.


--
-- Data for Name: equipment; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.equipment (name, category, supplier, description, id, created_at, photo) FROM stdin;
单杠	AEROBIC	奥瑞特	desc	2	2026-04-20 03:36:38.268936	\N
跑步机	AEROBIC	奥瑞特	\N	3	2026-04-20 04:09:02.122662	\N
篮球架	ACCESSORY	奥瑞特	\N	4	2026-04-20 04:09:41.366202	\N
\.


--
-- Data for Name: inspection; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.inspection (id, venue_id, inspector_id, current_location, status, description, created_at, inspection_time, photo_url, community_id) FROM stdin;
4	4	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.004721	[]	1
5	5	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.006776	[]	1
6	6	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.00901	[]	1
7	7	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.01115	[]	1
8	8	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.013259	[]	1
9	9	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.015043	[]	1
10	10	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.016828	[]	1
11	11	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.018529	[]	2
12	12	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.02029	[]	2
13	13	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.022007	[]	2
14	14	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.0239	[]	2
15	15	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.025702	[]	2
16	16	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.027575	[]	2
17	17	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.029588	[]	2
18	18	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.031375	[]	2
19	19	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.033303	[]	2
20	20	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.035079	[]	2
21	21	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.036862	[]	2
22	41	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.039866	[]	4
23	22	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.041971	[]	3
24	23	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.043779	[]	3
25	24	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.045798	[]	3
26	25	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.047515	[]	3
27	26	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.049302	[]	3
28	27	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.050969	[]	3
29	28	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.052705	[]	4
30	29	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.054426	[]	4
31	30	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.056361	[]	4
32	31	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.05823	[]	4
33	32	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.060131	[]	4
34	33	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.06207	[]	4
35	34	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.063914	[]	4
36	35	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.065622	[]	4
37	36	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.067318	[]	4
38	37	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.068951	[]	4
39	38	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.070651	[]	4
40	39	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.072329	[]	4
41	40	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.074125	[]	4
42	42	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.07586	[]	4
43	43	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.077887	[]	4
44	44	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.079975	[]	4
45	45	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.081813	[]	4
46	46	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.083492	[]	4
47	47	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.085262	[]	4
48	48	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.086963	[]	4
49	49	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.088912	[]	4
50	50	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.09062	[]	4
51	51	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.092427	[]	5
52	52	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.094351	[]	5
53	53	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.09657	[]	5
54	54	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.098519	[]	5
55	55	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.10027	[]	5
56	56	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.102018	[]	5
57	57	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.103728	[]	5
58	58	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.105467	[]	5
59	59	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.107335	[]	5
60	60	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.109202	[]	5
61	61	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.110962	[]	5
62	62	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.112855	[]	5
63	63	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.114639	[]	5
64	64	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.116433	[]	5
65	65	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.118139	[]	5
66	66	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.119896	[]	5
67	67	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.121501	[]	5
68	68	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.123294	[]	5
69	69	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.125124	[]	5
70	70	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.126902	[]	6
71	71	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.128981	[]	6
72	72	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.130886	[]	6
73	73	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.132596	[]	6
74	74	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.134535	[]	6
75	75	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.136269	[]	6
76	76	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.138083	[]	6
77	77	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.140396	[]	6
78	78	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.142277	[]	6
79	79	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.144031	[]	6
80	80	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.145969	[]	6
81	81	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.1481	[]	7
82	82	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.150012	[]	7
83	83	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.151797	[]	7
84	84	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.15354	[]	7
85	85	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.155286	[]	7
86	107	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.157107	[]	12
87	86	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.15909	[]	7
88	87	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.160925	[]	7
89	88	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.162904	[]	8
90	89	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.164886	[]	8
91	90	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.166757	[]	8
92	91	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.168501	[]	8
93	92	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.170493	[]	8
94	93	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.172269	[]	8
95	94	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.174258	[]	8
96	95	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.176167	[]	8
97	96	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.178148	[]	9
3	3	2		2	\N	2026-04-01 00:00:00	2026-04-07 02:39:50.05842	["http://localhost:8000/static/uploads/inspection/2026/04/detail_item_07023948_aa251010.jpg"]	1
98	97	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.180132	[]	9
99	98	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.182236	[]	9
100	99	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.183991	[]	9
101	100	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.185763	[]	10
102	101	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.187512	[]	10
103	102	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.189277	[]	10
104	103	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.191236	[]	10
105	104	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.193534	[]	11
106	105	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.1955	[]	11
107	106	2		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.197262	[]	11
108	108	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.199744	[]	13
109	109	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.201457	[]	13
110	110	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.203189	[]	13
111	111	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.204924	[]	13
112	112	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.206732	[]	13
113	113	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.208461	[]	13
114	114	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.210284	[]	13
115	115	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.212271	[]	13
116	116	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.214023	[]	13
117	117	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.215831	[]	13
118	118	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.217571	[]	13
119	119	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.219313	[]	14
120	120	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.221094	[]	14
121	121	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.222886	[]	14
122	122	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.224762	[]	14
123	123	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.226509	[]	14
124	124	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.228422	[]	14
125	125	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.230235	[]	14
126	126	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.231979	[]	14
127	127	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.233723	[]	14
128	128	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.235471	[]	14
129	129	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.237188	[]	14
130	130	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.239046	[]	15
131	131	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.240961	[]	15
132	132	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.242903	[]	15
133	133	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.245474	[]	16
134	134	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.247583	[]	16
135	135	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.249574	[]	16
136	136	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.25137	[]	16
137	137	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.253288	[]	16
138	138	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.255219	[]	16
139	139	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.257167	[]	16
140	140	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.258931	[]	16
141	141	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.260776	[]	16
142	142	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.262567	[]	16
143	143	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.2644	[]	16
144	144	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.266301	[]	16
145	145	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.268123	[]	16
146	146	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.27012	[]	16
147	147	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.271949	[]	18
148	148	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.274047	[]	18
149	149	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.276043	[]	18
150	150	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.278018	[]	18
151	151	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.280147	[]	18
152	152	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.281955	[]	18
153	153	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.283737	[]	18
154	154	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.285647	[]	19
155	155	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.287422	[]	19
156	156	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.289197	[]	19
157	157	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.290967	[]	19
158	158	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.29284	[]	19
159	159	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.294581	[]	19
160	160	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.296399	[]	19
161	161	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.298498	[]	19
162	162	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.300275	[]	19
163	163	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.301985	[]	19
164	164	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.303782	[]	19
165	165	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.305502	[]	19
166	166	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.307333	[]	20
167	167	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.309153	[]	20
168	168	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.310986	[]	20
169	169	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.312847	[]	20
170	170	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.314649	[]	20
171	171	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.316445	[]	20
172	172	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.318277	[]	20
173	173	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.320069	[]	21
174	174	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.321884	[]	21
175	175	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.323686	[]	21
176	176	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.325518	[]	21
177	177	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.328057	[]	22
178	178	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.329867	[]	22
179	179	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.331566	[]	22
180	180	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.333304	[]	23
181	181	3		0	\N	2026-04-01 00:00:00	2026-04-07 02:28:25.335143	[]	23
1	1	2		1	\N	2026-04-01 00:00:00	2026-04-07 02:29:11.310671	["http://localhost:8000/static/uploads/inspection/2026/04/detail_item_07022908_ed4e023f.jpg"]	1
2	2	2		1	\N	2026-04-01 00:00:00	2026-04-07 02:38:33.05974	["http://localhost:8000/static/uploads/inspection/2026/04/detail_item_07023831_51e1d576.jpg"]	1
\.


--
-- Data for Name: maintenance; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.maintenance (id, inspection_id, venue_id, equipment_id, equipment_name, maintainer_id, issue_type, status, action_taken, before_photo, completion_photo, create_user_id, created_at, reported_at, finished_at, is_urgent, completion_remark, update_at) FROM stdin;
1	\N	206	1	\N	\N	GROUND_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_09033508_13b7a96e.jpg", "http://localhost:8000/static/uploads/repair/2026/04/site_overall_09033511_514c8d6e.jpg"]	[]	1	2026-04-09 03:35:17.408321	2026-04-09 03:35:17.408319	\N	f	\N	\N
2	\N	5	1	\N	\N	EQUIPMENT_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_09034629_7587bf6a.jpg"]	[]	1	2026-04-09 03:46:33.485416	2026-04-09 03:46:33.485413	\N	f	\N	\N
3	\N	4	1	\N	\N	GROUND_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_09034804_8eccc48a.jpg"]	[]	1	2026-04-09 03:48:15.422473	2026-04-09 03:48:15.42247	\N	f	\N	\N
4	\N	338	1	\N	\N	GROUND_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_09035547_96aabdaa.jpg"]	[]	1	2026-04-09 03:55:52.813835	2026-04-09 03:55:52.813834	\N	f	\N	\N
5	\N	13	4	篮球架	4	CLEANLINESS_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_20041302_3cb6ba2e.jpg"]	[]	1	2026-04-20 04:13:08.424299	2026-04-20 04:13:08.424297	\N	f	\N	2026-04-20 04:13:08.424614
6	\N	14	2	单杠	4	GROUND_ISSUE	0	\N	["http://localhost:8000/uploads/repair/2026/04/site_overall_20041858_d0527c87.jpg"]	[]	1	2026-04-20 04:19:02.502077	2026-04-20 04:19:02.502075	\N	f	\N	2026-04-20 04:19:02.502386
7	\N	7	2	单杠	4	OTHER_ISSUE	0	\N	["http://localhost:8000/static/uploads/repair/2026/04/site_overall_20042439_70de68da.png"]	[]	1	2026-04-20 04:24:42.957088	2026-04-20 04:24:42.957086	\N	f	\N	2026-04-20 04:24:42.957429
\.


--
-- Data for Name: maintenance_history; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.maintenance_history (id, maintaine_id, maintainer_id, maintainer_name, action, create_user_id, created_at) FROM stdin;
\.


--
-- Data for Name: street; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.street (id, name, contact_name, contact_phone, region) FROM stdin;
1	龙华街道	-	-	LH
2	观澜街道	-	-	LH
3	民治街道	-	-	LH
4	福城街道	-	-	LH
5	观湖街道	-	-	LH
6	大浪街道	-	-	LH
\.


--
-- Data for Name: user; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public."user" (id, full_name, phone, role, password, is_disabled, login_retry_count, openid, created_at) FROM stdin;
1	admin	18938919024	ADMIN	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.912712
2	巡查1	18938919021	INSPECTOR	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.915637
3	巡查2	18938919022	INSPECTOR	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.917658
4	维修1	18938919023	MAINTAINER	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.919687
5	维修2	18938919025	MAINTAINER	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.921697
6	甲方	18938919026	CLIENT_ADMIN	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.923988
7	群众	18938919027	CITIZEN	$2b$12$x9UJN3WHPpS6POQ8jJY5OOjyWgk6xKuMvyrLE0tEBc7AB4VIvM3mi	f	0	\N	2026-04-07 02:26:15.926323
\.


--
-- Data for Name: user_data_permissions; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.user_data_permissions (id, user_id, scope_type, scope_id, scope_name) FROM stdin;
1	2	STREET	1	龙华街道
2	3	STREET	2	观澜街道
3	4	STREET	1	龙华街道
4	5	STREET	2	观澜街道
5	6	STREET	1	龙华街道
\.


--
-- Data for Name: venue; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.venue (id, management_unit, owner_unit, floor_type, address, contact_name, contact_phone, warranty_period, area_size, venue_type, equipment_type, equipment_brand, venue_count, equipment_count, install_date, community_id, name, photo_url, district_id, street_id, equipment_detail, damage_description, damage_count) FROM stdin;
4	\N	\N	拼装地板	瑞丰花园	钟家燕	13420955222	\N	100	健身路径	\N	好家庭	1	12	2017-09-01	1	瑞丰花园	[]	LH	1	腹肌板，三位扭腰器*2，揉推按摩器，漫步机，四级压腿按摩器，骑马器，上肢牵引器，告示牌，肋木架，太极揉推器，腰背按摩器	正常	0
5	\N	\N	安全地垫	水斗老围二区	钟家燕	13420955222	\N	55	健身路径	\N	（好家庭）	1	4	未知	1	水斗老围二区	[]	LH	1	太极揉推器，腰背按摩器，揉推按摩器，展背器	安全地垫破3㎡	1
6	\N	\N	安全地垫	水斗老围一区	钟家燕	13420955222	\N	140	健身路径	\N	好家庭	1	22	2016-01-01	1	水斗老围一区	[]	LH	1	上肢牵引器*2，三位扭腰器*2，骑马器*2，伸背器，告示牌*2，肋木架，双杆，二联单杠，伸背器*2，腹肌板，蹬力器，漫步机*2，腰背按摩器，太极揉推器，按摩揉推器，四级压腿按摩器	安全地垫破损5㎡，三位扭腰器转盘缺2个，太空漫步机盖帽缺2个	3
7	\N	\N	拼装地板	新阳丽舍	钟家燕	13420955222	\N	90	健身路径	\N	（好家庭）	1	15	未知	1	新阳丽舍	[]	LH	1	太极揉推器，双杆，腹肌板，展背器，腰背按摩器，骑马器，漫步机，告示牌，三位扭腰器，单杆，肋木架*2，上肢牵引器，揉推按摩器，伸背器，象棋桌	骑马器限位损坏，漫步机轴承损坏，象棋桌桌面破损	3
8	\N	\N	安全地垫	荔苑山庄	钟家燕	13420955222	\N	450	健身路径	\N	（好家庭） 奥瑞特	1	38	未知	1	荔苑山庄	[]	LH	1	上肢牵引器*2，弹振压腿按摩器，双杆，腰背按摩器，单杠，揉推按摩器，太极揉推器*2，肋木架，三位扭腰器，漫步机*2，天梯，告示牌（奥瑞特） 告示牌*2，四人蹬力器，三人骑马器，上肢牵引器，二人蹬力器*2，腰背按摩器*2，骑马器*2，四级压腿按摩器*3，漫步机*2，双位腹肌板*2，双人扭腰器*2，环形扭腰步道，太极揉推器*2（好家庭）	正常	0
9	\N	\N	硅pu	利坚城工业园	钟家燕	13420955222	\N	/	篮球场	\N	/	1	1	2013-01-01	1	利坚城工业园	[]	LH	1	篮球场	篮筐缺失1个，篮网缺失1个	0
10	\N	\N	拼装地板	油松派出所营房二楼天台	钟家燕	13420955222	\N	/	篮球场	\N	杰帝奇	1	1	2023-01-01	1	油松派出所营房二楼天台	[]	LH	1	篮球场	正常	2
11	\N	\N	拼装地板	龙华街道胜立工业园	李琳	15813801799	\N	110	健身路径	\N	好家庭	1	12	\N	2	龙华街道胜立工业园	[]	LH	1	告示牌，棋牌桌，漫步机，骑马器，三位扭腰器，腹肌板，展背器，四级压腿按摩器，太极揉推器，按摩揉推器，蹬力器，肋木架，	正常	0
12	\N	\N	拼装地板	龙华街道清湖新村2（清湖社区公园）	李琳	15813801799	\N	120	健身路径	\N	奥瑞特	1	12	2008年	2	龙华街道清湖新村2（清湖社区公园）	[]	LH	1	双位上肢牵引器，腹背屈伸划船器，自重式前推下拉训练器，双位漫步机，双位骑马器，双位蹬力器，双位扭腰器，背肌腹肌训练器，太极揉推器，双位压腿按摩器，告示牌，组合训练器	蹬力器限位损坏，骑马器限位损坏	2
13	\N	\N	水泥地	龙华街道清湖新村3（清湖 社区公园）	李琳	15813801799	\N	80	健身路径	\N	杂牌	1	10	2008年	2	龙华街道清湖新村3（清湖 社区公园）	[]	LH	1	告示牌*2，天梯，单杠*2，腿部按摩器，蹬力器，三位扭腰器，双人骑马器，双位腹肌板。	正常	0
14	\N	\N	水泥地	龙华街道清湖新村（村口处）	李琳	15813801799	\N	300	健身路径	\N	好家庭	1	14	2009年	2	龙华街道清湖新村（村口处）	[]	LH	1	五边形组合体测中心，告示牌，上肢肩关节训练器，深蹲提踵训练器，天梯，单杠组合训练器，棋牌桌*2，双杠，战绳，竞赛车*2，高啦推举训练器，扩胸划船训练器	多处屏幕失灵	1
15	\N	\N	\N	龙华街道清湖老村篮球场	李琳	15813801799	\N	500	篮球场	\N	/	1	1	2008年	2	龙华街道清湖老村篮球场	[]	LH	1	篮球场	地面破损严重，篮网缺失*1	2
16	\N	\N	\N	龙华街道花半里	李琳	15813801799	\N	180	健身路径	\N	好家庭	1	14	2009年	2	龙华街道花半里	[]	LH	1	告示牌，腹肌板，肋木架，双杆，三位扭腰器，二连单杠，上肢牵引器，按摩揉推器，太极揉推器，骑马器，象棋桌，展背器，腰背按摩器，伸背器	三位扭腰器转盘缺1，骑马器限位损坏	2
17	\N	\N	\N	清湖工业城20栋旁	李琳	15813801799	\N	80	健身路径	\N	好家庭	1	12	2017-01-01	2	清湖工业城20栋旁	[]	LH	1	象棋桌，按摩揉推器，腰背按摩器，肋木架，上肢牵引器，三位扭腰器，告示牌，压腿按摩器，太极揉推器，骑马器，漫步机，腹肌板	骑马器限位损坏	1
18	\N	\N	\N	硅谷动力清湖园	李琳	15813801799	\N	40	健身路径	\N	天行健	1	6	2018年	2	硅谷动力清湖园	[]	LH	1	蹲举训练器，坐式蹬力训练器，腿部训练器，坐式前推训练器，推举训练器，扩胸训练器	多件器材限位损坏，补漆	2
19	\N	\N	\N	清湖社区硅谷动力清湖园	李琳	15813801799	\N	500	篮球架	\N	/	1	1	2019年	2	清湖社区硅谷动力清湖园	[]	LH	1	篮球场	两副，（报废一个，一个篮网破损）	1
20	\N	\N	\N	龙华街道硅谷动力清湖园	李琳	15813801799	\N	40	健身路径	\N	桂宇星	1	11	2015年	2	龙华街道硅谷动力清湖园	[]	LH	1	告示牌，跷跷板，象棋桌，腰背按摩器，钟摆器，压腿架，平步机，双人手臂支持，漫步机，太极揉推器*2	器材老旧	0
21	\N	\N	拼装地板	龙华街道富安娜工业园篮球+健身路径	李琳	15813801799	\N	80	篮球场+健身路径	\N	奥瑞特	1	10	2008年	2	龙华街道富安娜工业园篮球+健身路径	[]	LH	1	告示牌，腰背按摩器，漫步机*2，三位扭腰器，太极揉推器，展背器，多功能训练器，斜躺健身车，仰卧起坐训练器	仰卧起坐训练器盖帽缺*3	1
41	\N	\N	EPDM	龙华街道世纪华庭1	李景龙	13424431439	\N	60	篮球场	\N	未知	1	1	2016年	4	龙华街道世纪华庭1	[]	LH	1	篮球场	地面破旧，剩半场	1
2	\N	\N	硅pu	水斗新围村1	钟家燕	13420955222	\N	90	健身路径	\N	（杂牌）	1	4	未知	1	水斗新围村1	["http://localhost:8000/static/uploads/inspection/2026/04/site_overall_07023828_646f3aa3.jpg"]	LH	1	腰背按摩器，双人蹬力器，伸背器，三位扭腰器	地面破损	1
3	\N	\N	硅pu	香提雅苑	钟家燕	13420955222	\N	66	健身路径	\N	好家庭	1	11	2017-09-01	1	香提雅苑	["http://localhost:8000/static/uploads/inspection/2026/04/site_overall_07023946_78559f38.jpg"]	LH	1	肋木架，上肢牵引器，腹肌训练板，告示牌，太极推手器，太空漫步机，三位扭腰器，腰背按摩器，按摩揉推器，骑马器，四级压腿器	拼装地板缺4.5㎡	1
22	\N	\N	水泥地	龙华新区民清路第二住宅小区	罗云青	13809899348	\N	30	健身路径	\N	好家庭	1	6	2017年	3	龙华新区民清路第二住宅小区	[]	LH	1	棋牌桌，上肢牵引器，按摩揉推器，腰背按摩器，太极揉推器，四级压腿按摩器	因加装电梯，拆除部分	0
23	\N	\N	拼装地板	龙华文化艺术中心	罗云青	13809899348	\N	250	智能健身房	\N	好家庭	1	21	2020年投放	3	龙华文化艺术中心	[]	LH	1	高啦推举训练器，上肢屈伸训练器，腿部屈伸训练器，推胸划船训练器，五边形体侧系统，腹背肌训练器，深蹲提踵训练器，立式健身车，立式手摇车，智能竞赛车*3，战绳，背肌训练器，腹肌板，棋牌桌*4，双杠，单杠	场馆升级改造中，无人使用。	0
24	\N	\N	EPDM	龙华街道龙华广场	罗云青	13809899348	\N	200	健身路径	\N	好家庭	1	27	2010年	3	龙华街道龙华广场	[]	LH	1	漫步机*2，太极揉推器*3，腰背按摩器*2，蹬力器，天梯*2，二连单杠，三位扭腰器**3，肋木架*2，告示牌，上肢牵引器，双杠*3，弹振压腿器，展背器，平步机，角力器，腹肌板，四级压腿按摩器	腰背按摩器损坏，弹振压腿器盖帽缺失，腹肌板盖帽缺失，上肢牵引器轴承损坏	4
25	\N	\N	安全地垫	龙华街道东源阁	罗云青	13809899348	\N	120	健身路径	\N	好家庭	1	13	2014年	3	龙华街道东源阁	[]	LH	1	告示牌，漫步机，弹振压腿器，按摩揉推器，展背器，蹬力器，太极揉推器，腰背按摩器，双位腹肌板，二连单杠，多功能训练器，棋牌桌，三位扭腰器	腰背按摩器转盘轴承损坏	1
26	\N	\N	拼装地板	富士康C区	罗云青	13809899348	\N	120	健身路径	\N	奥瑞特	1	30	2022-01-01	3	富士康C区	[]	LH	1	棋牌桌，展背器*2，腰背按摩器*2，三联单杠*2，背肌训练器，双杠*2，蹬力器*2，腹肌板，告示牌*2，四级压腿按摩器，腰背按摩器，太极揉推器，漫步机，背腹训练器，椭圆机*2，蹬力器，组合训练器，棋牌桌，弹振压腿器，漫步机*2，钟摆器，平步机，	上下肢训练器限位损坏，四级压腿按摩器滚轮损坏，腰背按摩器限位损坏	3
27	\N	\N	水泥地	富士康J区	罗云青	13809899348	\N	100	健身路径	\N	奥瑞特	1	31	\N	3	富士康J区	[]	LH	1	伸背器*2，三位扭腰器*3，腰背按摩器，展背器*，上肢牵引器，三联单杠，双杠*3，压腿器*2，双位腹肌板*2，上肢牵引器，肋木架，双杠，二连单杠*2，骑马器，按摩揉推器，漫步机，坐拉器，引体向上训练器，告示牌，双位腹肌板，展背器，骑马器，腰背按摩器，	按摩揉推器转盘损坏	1
28	\N	\N	水泥地	龙华街道三联二区九巷	李景龙	13424431439	\N	30	健身路径	\N	杂牌	1	4	2016年	4	龙华街道三联二区九巷	[]	LH	1	骑马器，漫步机，太极揉推器*2	漫步机整件松动	1
29	\N	\N	水泥地	龙华街道美丽家园北区	李景龙	13424431439	\N	80	健身路径	\N	好家庭未知	1	10	2014年	4	龙华街道美丽家园北区	[]	LH	1	告示牌，漫步机，按摩揉推器，展背器，骑马器，肋木架，太极揉推器，腰背按摩器，伸背器，上肢牵引器，	正常	0
30	\N	\N	安全地垫	龙华街道美丽家园南区	李景龙	13424431439	\N	120	健身路径	\N	体之杰+好家庭	1	16	2016年	4	龙华街道美丽家园南区	[]	LH	1	告示牌，展背器，骑马器，坐推器，做拉器，太极揉推器，压腿器，腹肌板，二连单杠，漫步机，四人蹬力器（体之杰）   三位扭腰器，腹肌板，棋牌桌，双杠，二连单杠（好家庭）	漫步机摆腿缺失（体之杰）	1
31	\N	\N	/	绿茵华庭，龙华区龙观大道381号	李景龙	13424431439	\N	/	儿童滑梯	\N	未知	1	1	2022-12-09	4	绿茵华庭，龙华区龙观大道381号	[]	LH	1	儿童滑梯	正常	0
32	\N	\N	EPDM	龙华街道绿茵华庭	李景龙	13424431439	\N	100	健身路径	\N	好家庭	1	12	2013年	4	龙华街道绿茵华庭	[]	LH	1	二连单杠，肋木架，三位扭腰器，漫步机*2，告示牌，双杠，四级压腿按摩器，骑马器，太极揉推器*2，腰背按摩器，	正常	0
33	\N	\N	拼装地板	龙华街道锦锈御园3栋旁	李景龙	13424431439	\N	70	健身路径	\N	未知（奥瑞特）	1	11	2016年	4	龙华街道锦锈御园3栋旁	[]	LH	1	环形上肢协调功能训练器，踝关节屈伸训练器*2，骑行式下肢训练器*2，手摇式上肢训练器*2，平行杠步态训练器，肩关节回旋/前臂腕关节训练器，背部伸展/腕关节/握力训练器，肩梯/上肢协调功能训练器。	踝关节屈伸训练器踏板损坏，地面损坏11平	2
34	\N	\N	拼装地板	龙华街道锦锈御园2栋旁	李景龙	13424431439	\N	80	健身路径	\N	好家庭	1	14	2014年	4	龙华街道锦锈御园2栋旁	[]	LH	1	三位扭腰器，骑马器，展背器，漫步机，告示牌，太极揉推器，腰背按摩器，蹬力器，肋木架，上肢牵引器，按摩揉推器，四级压腿按摩器，腹肌板，棋牌桌	腹肌板腐蚀严重	1
35	\N	\N	拼装地板	高翔路28号锦绣御园8B栋2单元架空层	李景龙	13424431439	\N	80	健身路径	\N	奥瑞特	1	11	2023年	4	高翔路28号锦绣御园8B栋2单元架空层	[]	LH	1	多功能锻炼器，蹬力压腿训练器，自重式下压训练器，三位扭腰器，展背器，骑马器，告示牌，太极揉推器，上肢牵引器，腰背按摩器，漫步机	正常	0
36	\N	\N	地砖	龙华街道狮头岭新村西区	李景龙	13424431439	\N	40	健身路径	\N	杂牌	1	4	2015年	4	龙华街道狮头岭新村西区	[]	LH	1	太极揉推器，骑马？平步机组合，做拉器，漫步机，	坐拉器缺失一半，漫步机摆腿缺失3，骑马平步机未预埋安装	3
37	\N	\N	水泥地	龙华街道狮头岭东区	李景龙	13424431439	\N	60	健身路径	\N	杂牌	1	5	2015年	4	龙华街道狮头岭东区	[]	LH	1	双人骑马器，双人坐拉器，太极揉推器*2，漫步机	坐拉器拉臂缺失	1
38	\N	\N	拼装地板	龙华街道东和花园	李景龙	13424431439	\N	150	健身路径	\N	好家庭+奥瑞特	1	14	2017年	4	龙华街道东和花园	[]	LH	1	二连单杠，双杠，三位扭腰器，按摩揉推器，展背器，漫步机，太极揉推器，四级压腿按摩器，腰背按摩器，蹬力器，骑马器，上肢牵引器，腹肌板，肋木架	正常	0
39	\N	\N	拼装地板	龙华街道世纪华庭	李景龙	13424431439	\N	70	健身路径	\N	好家庭	1	14	2016年	4	龙华街道世纪华庭	[]	LH	1	告示牌，四级压腿按摩器，棋牌桌，展背器，蹬力器，肋木架，按摩揉推器，上肢牵引器，漫步机，太极揉推器，腰背按摩器，三位扭腰器，骑马器，腹肌板，	蹬力器限位损坏，骑马器限位损坏，四级压腿按摩器盖帽缺失	3
40	\N	\N	/	龙华街道世纪华庭1	李景龙	13424431439	\N	/	健身路径	\N	杂牌	1	10	2010年	4	龙华街道世纪华庭1	[]	LH	1	旋转轮，双杠，腰背按摩器，漫步机，三人健腰器，坐推训练器，坐拉训练器，坐蹬训练器，伸腰器，太极轮	整体老旧，地面老旧，实用率低	0
42	\N	\N	拼装地板	龙华街道美丽AAA花园	李景龙	13424431439	\N	50	健身路径	\N	杂牌	1	7	2012年	4	龙华街道美丽AAA花园	[]	LH	1	太极揉推器，三位扭腰器，漫步机*2，双人骑马器，二连单杠，双杠，	器材老旧，地面整体破损严重	2
43	\N	\N	拼装地板	龙华街道美丽AAA花园	李景龙	13424431439	\N	100	篮球场	\N	未知	1	1	2011年	4	龙华街道美丽AAA花园	[]	LH	1	\N	正常	0
44	\N	\N	/	三联社区城市明珠花园小区	李景龙	13424431439	\N	/	健身路径	\N	好家庭	1	9	2018年	4	三联社区城市明珠花园小区	[]	LH	1	腹肌板，展背器，三位扭腰器，腰背按摩器，太极揉推器，骑马器，按摩揉推器，大转轮，告示牌	骑马器限位损坏	1
45	\N	\N	/	龙华街道城市明珠花园	李景龙	13424431439	\N	/	健身路径	\N	奥瑞特	1	13	2016年	4	龙华街道城市明珠花园	[]	LH	1	告示牌，展背器，转手器，腰背按摩器，多功能锻炼器，上肢牵引器，漫步机，三位扭腰器，骑马器，太极揉推器，二连单杠，双杠，自重式下压训练器	正常	0
46	\N	\N	拼装地板	美丽365花园	李景龙	13424431439	\N	80	篮球场	\N	未知	1	1	2017年	4	美丽365花园	[]	LH	1	\N	正常	0
47	\N	\N	拼装地板	美丽365花园	李景龙	13424431439	\N	50	儿童滑梯	\N	未知	1	1	2022-03-08	4	美丽365花园	[]	LH	1	\N	正常	0
48	\N	\N	安全地垫	龙华街道美丽365花园	李景龙	13424431439	\N	60	健身路径	\N	好家庭	1	13	2008年	4	龙华街道美丽365花园	[]	LH	1	坐式蹬力训练器，蹲举训练器，漫步机，展背器，按摩揉推器，肋木架，棋牌桌，告示牌*2，太极揉推器，四级压腿按摩器，腰背按摩器，腹肌板	蹲举训练器限位损坏，按摩揉推器转盘缺失，腹肌板腐蚀严重	3
49	\N	\N	安全地垫	龙华街道美丽365花园2	李景龙	13424431439	\N	60	健身路径	\N	杂牌	1	3	2016年	4	龙华街道美丽365花园2	[]	LH	1	二连单杠，双杠，告示牌	双杠腐蚀割手	1
50	\N	\N	安全地垫	美丽365花园	李景龙	13424431439	\N	60	健身路径	\N	奥瑞特	1	11	2021年	4	美丽365花园	[]	LH	1	仰卧起坐训练器，三位扭腰器，蹬力器，展背器，腰背按摩器，太极揉推器，按摩揉推器，漫步机，弹振压腿器，多功能训练器，告示牌	腰背按摩器滚轮卡住	1
51	\N	\N	安全地垫	龙华街道桦润馨居	黄小姐	13590283306	\N	110	健身路径	\N	好家庭	1	19	2014年	5	龙华街道桦润馨居	[]	LH	1	告示牌，二连单杠，上肢牵引器，肋木架，伸背器，太极揉推器*3，按摩揉推器，骑马器，坐蹬器，展背器，腹肌板，坐拉器，双杆，腰背按摩器，漫步机，三位扭腰器，棋牌桌	地面破损110平米，三位扭腰器转盘腐蚀*1，腹肌板 腐蚀穿孔，太极揉推器盖帽损坏	4
52	\N	\N	EPDM	龙华街道乐景花园	黄小姐	13590283306	\N	70	健身路径	\N	好家庭	1	11	2016年	5	龙华街道乐景花园	[]	LH	1	告示牌，上肢牵引器，钟摆器，太极揉推器*2，腰背按摩器，四级压腿按摩器，骑马器，腹肌板，三位扭腰器，双杠	正常	0
53	\N	\N	安全地垫	龙华街道金侨花园	黄小姐	13590283306	\N	100	健身路径	\N	奥瑞特	1	14	2009年	5	龙华街道金侨花园	[]	LH	1	肋木架，上肢牵引器，漫步机，三位扭腰器，双联平步机，伸背器，按摩揉推器，二连单杠，蹬力器，告示牌，棋牌桌，弹振压腿器，腹肌板，太极揉推器	双联平步机损坏，蹬力器轴承损坏护盖缺失*4，整体地面破损100平米，太极揉推器损坏	4
54	\N	\N	硅PU	龙华街道金侨花园篮球场	黄小姐	13590283306	\N	500	篮球场	\N	/	1	1	2011年	5	龙华街道金侨花园篮球场	[]	LH	1	篮球场	地面破旧	1
55	\N	\N	硅PU	优品建筑	黄小姐	13590283306	\N	500	篮球架	\N	/	1	1	2019年	5	优品建筑	[]	LH	1	篮球架	正常	0
56	\N	\N	拼装地板	龙华街道优品建筑	黄小姐	13590283306	\N	70	健身路径	\N	奥瑞特	1	16	2017年	5	龙华街道优品建筑	[]	LH	1	弹振压腿器，告示牌，三位扭腰器，太极揉推器，晃板扭腰器，手部腿部按摩器，腰背按摩器，角力器，蹬力器，上肢牵引器，肋木架，棋牌桌，仰卧起坐训练器，漫步机，椭圆机，骑马器，单杠	蹬力器限位损坏，椭圆机轴承损坏，地面缺10平米	3
57	\N	\N	拼装地板	龙华街道金碧世家	黄小姐	13590283306	\N	80	健身路径	\N	好家庭	1	13	2011年	5	龙华街道金碧世家	[]	LH	1	告示牌，上肢牵引器，肋木架，蹬力器，漫步机，三位扭腰器，展背器，太极揉推器，四级压腿按摩器，按摩揉推器，骑马器，腹肌板，腰背按摩器，	正常	0
58	\N	\N	拼装地板	龙华街道富通天骏	黄小姐	13590283306	\N	60	健身路径	\N	奥瑞特	1	8	2009年	5	龙华街道富通天骏	[]	LH	1	告示牌，天梯，肋木架，二连单杠，漫步机，弹振压腿器，俯卧撑训练器，双杠	正常	0
59	\N	\N	拼装地板	龙华街道大信花园	黄小姐	13590283306	\N	250	健身路径	\N	好家庭	1	20	2008年	5	龙华街道大信花园	[]	LH	1	双位腹肌板，四人蹬力器，三位扭腰器，三人骑马器，扭腰步道，告示牌，三联单杠，肋木架，双杠，上肢牵引器，攀爬梯，腰背按摩器，按摩揉推器，太极揉推器，展背器，骑马器，腹肌板，伸背器，天梯，组合训练器	三联单杠盖帽处腐蚀，部分器材老旧。	2
60	\N	\N	拼装地板	龙华街道南国丽园	黄小姐	13590283306	\N	210	健身路径	\N	好家庭	1	14	2010年	5	龙华街道南国丽园	[]	LH	1	腰背按摩器，三位扭腰器，骑马器，肋木架，蹬力器，上肢牵引器，天梯，太极揉推器，展背器，漫步机，四级压腿按摩器，腹肌板，按摩揉推器*2	腰背按摩器滚轮卡住	1
61	\N	\N	EPDM	中环花园	黄小姐	13590283306	\N	210	健身路径	\N	好家庭	1	18	2017年	5	中环花园	[]	LH	1	肋木架*2，天梯，二连单杠，漫步机，腹肌板，按摩揉推器，钟摆器，上肢牵引器，压腿器，告示牌，太极揉推器，骑马器*2，四级压腿按摩器，三位扭腰器，腰背按摩器，四人蹬力器	正常	0
62	\N	\N	拼装地板	景龙社区工作站	黄小姐	13590283306	\N	130	健身路径	\N	奥瑞特	1	23	2017年	5	景龙社区工作站	[]	LH	1	三位扭腰器，告示牌，肋木架*2，上肢牵引器，柔韧训练器，棋牌桌，双杠，漫步机，弹振压腿按摩器，蹬力器，腰背按摩器*2，二连单杠，三位扭腰器，按摩揉推器，天梯，太极揉推器，四级压腿按摩器，骑马器，腹肌板，	腰背按摩器需更换总成，漫步机轴承损坏	2
63	\N	\N	拼装地板	东华明珠园	黄小姐	13590283306	\N	60	健身路径	\N	奥瑞特	1	8	2017年	5	东华明珠园	[]	LH	1	二连单杠，天梯，肋木架，俯卧撑训练器，双杠，告示牌，漫步机，弹振压腿器，	漫步机轴承损坏	1
64	\N	\N	拼装地板	龙华区龙华街道东华明珠园	黄小姐	13590283306	\N	54	健身路径	\N	天行健	1	7	2019年	5	龙华区龙华街道东华明珠园	[]	LH	1	拉力训练器，扩胸训练器，推举训练器，腿部训练器，坐式蹬力训练器，坐推训练器	正常	0
65	\N	\N	拼装地板	龙华街道东华明珠园	黄小姐	13590283306	\N	100	健身路径	\N	好家庭	1	14	2017年	5	龙华街道东华明珠园	[]	LH	1	漫步机，三位扭腰器，骑马器，腹肌板，二连单杠，双杠，腰背按摩器，展背器，手部腿部按摩器，太极揉推器，告示牌，上肢牵引器，伸背器，肋木架	手部腿部按摩器转盘缺*1，三位扭腰器转盘腐蚀*1，双杠盖帽缺*1	3
66	\N	\N	拼装地板	景龙社区丹枫雅苑	黄小姐	13590283306	\N	120	健身路径	\N	好家庭	1	11	2017年	5	景龙社区丹枫雅苑	[]	LH	1	告示牌，四级压腿按摩器，太极揉推器，按摩揉推器，腰背按摩器，上肢牵引器，肋木架，三位扭腰器，漫步机，腹肌板，骑马器，	正常	0
67	\N	\N	拼装地板	龙华街道景龙新村	黄小姐	13590283306	\N	70	健身路径	\N	好家庭	1	14	2010年	5	龙华街道景龙新村	[]	LH	1	上肢牵引器，肋木架，蹬力器，三位扭腰器，太极揉推器，告示牌，漫步机，棋牌桌，骑马器，腹肌板，伸背器，四级压腿按摩器，按摩揉推器，腰背按摩器，	整体地面破旧，蹬力器限位损坏*2，按摩揉推器转盘损坏，骑马器限位损坏，棋牌桌凳子缺*3	5
68	\N	\N	安全地垫	龙华街道景华新村	黄小姐	13590283306	\N	100	健身路径	\N	杂牌未知	1	6	2012年	5	龙华街道景华新村	[]	LH	1	腰背按摩器*2，太极揉推器*2，蹬力器，扭腰器	太极揉推器转盘缺失	1
69	\N	\N	拼装地板	华昱苑3C单元旁	黄小姐	13590283306	\N	80	健身路径	\N	天行健	1	7	2018年	5	华昱苑3C单元旁	[]	LH	1	告示牌，扩胸训练器，坐式前推训练器，推举训练器，坐式蹬力训练器，蹲举训练器，腿部训练器	地面破损2平米，护盖缺失*4，护盖损坏*5，扩胸训练器，坐式前推训练器，坐式蹬力训练器，蹲举训练器，限位损坏	7
70	\N	\N	拼装地板	龙华街道金雍阁二期	-	15818546857	\N	100	健身路径	\N	好家庭	1	14	2010年	6	龙华街道金雍阁二期	[]	LH	1	上肢牵引器，蹬力器，按摩揉推器，腰背按摩器，太极揉推器，四级压腿按摩器，展背器，骑马器，棋牌桌，告示牌，腹肌板，三位扭腰器，肋木架，漫步机	蹬力器限位损坏，骑马器限位损坏，地面破损2平米	3
71	\N	\N	水泥地	龙华街道青年城邦园	-	15818546857	\N	60	健身路径	\N	奥瑞特	1	8	2017年	6	龙华街道青年城邦园	[]	LH	1	双杠，压腿器，棋牌桌，漫步机，太极揉推器，多功能锻炼器，腹肌板，平步机	正常	0
72	\N	\N	拼装地板	龙华街道花园新村	-	15818546857	\N	60	健身路径	\N	好家庭	1	5	2008年	6	龙华街道花园新村	[]	LH	1	告示牌，四级压腿按摩器，骑马器，太极揉推器，漫步机。	骑马器限位损坏	1
73	\N	\N	拼装地板	龙华街道花园新村1	-	15818546857	\N	80	健身路径	\N	奥瑞特	1	11	2010年	6	龙华街道花园新村1	[]	LH	1	骑马器，展背器，漫步机，三位扭腰器，告示牌，上肢牵引器，多功能锻炼器，腰背按摩器，转手器，太极揉推器，自重式下压训练器	正常	0
74	\N	\N	拼装地板	花园新村9栋	-	15818546857	\N	80	健身路径	\N	奥瑞特	1	11	2023年	6	花园新村9栋	[]	LH	1	告示牌，展背器，骑马器，自重式下压训练器，上肢牵引器，漫步机，多功能锻炼器，三位扭腰器，太极揉推器，腰背按摩器，蹬力压腿训练器	正常	0
75	\N	\N	拼装地板	龙华街道龙华公园	-	15818546857	\N	200	健身路径	\N	好家庭	1	18	2009年	6	龙华街道龙华公园	[]	LH	1	智能竞赛车（上肢),智能竞赛车（下肢），五边形组合体质测试器，推胸划船训练器，腿部屈伸训练器，深蹲提踵训练器，腹背肌训练器，高啦推举训练器，上肢屈伸训练器，上肢肩关节训练器，立式健身车，双位揉推器，双位钟摆器，双位漫步机，双位扭腰器，双杆*2，天梯	正常	0
76	\N	\N	硅PU	龙华街道龙华公园篮球场	-	15818546857	\N	500	篮球场	\N	/	1	1	2008年	6	龙华街道龙华公园篮球场	[]	LH	1	篮球场	正常	0
77	\N	\N	拼装地板	龙华街道隆源社区御筑轩	-	15818546857	\N	80	健身路径	\N	好家庭	1	14	2008年	6	龙华街道隆源社区御筑轩	[]	LH	1	三位扭腰器，双杠，二联单杠，腰背按摩器，太极揉推器，骑马器，伸背器，腹肌板，按摩揉推器，漫步机，上肢牵引器，腰背伸展器，肋木架，告示牌	正常	0
78	\N	\N	硅PU	龙华街道和平花园	-	15818546857	\N	500	篮球场	\N	/	1	1	2008年	6	龙华街道和平花园	[]	LH	1	篮球场	正常	0
79	\N	\N	拼装地板	龙华街道和平花园	-	15818546857	\N	100	健身路径	\N	好家庭	1	11	2010年	6	龙华街道和平花园	[]	LH	1	告示牌，上肢牵引器，肋木架，腰背按摩器，按摩揉推器，四级压腿按摩器，漫步机，三位扭腰器，骑马器，腹肌板，棋牌桌	正常	0
80	\N	\N	硅PU	龙园社区和平花园小区	-	15818546857	\N	500	篮球架	\N	/	1	1	2018年	6	龙园社区和平花园小区	[]	LH	1	篮球场	正常	0
81	\N	\N	拼装地板	华联社区新城市花园	杨先生	13510535992	\N	200	健身路径	\N	好家庭	1	12	2018年	7	华联社区新城市花园	[]	LH	1	告示牌，棋牌桌，三位扭腰器，大转轮，漫步机，腰背按摩器，按摩揉推器，太极揉推器，骑马器，腹肌板，展背器，四级压腿按摩器	骑马器限位损坏	1
82	\N	\N	硅PU	华联社区新城市花园	杨先生	13510535992	\N	500	篮球架	\N	/	1	1	2018年	7	华联社区新城市花园	[]	LH	1	篮球场	正常	0
83	\N	\N	拼装地板	华联社区新城市花园	杨先生	13510535992	\N	100	健身路径	\N	天行健	1	7	2018年	7	华联社区新城市花园	[]	LH	1	腿部训练器，坐式蹬力训练器，蹲举训练器，坐式前腿训练器，扩胸训练器，推举训练器，告示牌	正常	0
84	\N	\N	安全地垫	龙华街道老围新村	杨先生	13510535992	\N	50	健身路径	\N	好家庭	1	8	2010年	7	龙华街道老围新村	[]	LH	1	告示牌，骑马器，腹肌板，太极揉推器，展背器，四级压腿按摩器，腰背按摩器，按摩揉推器，	正常	0
85	\N	\N	安全地垫	龙华街道港侨新村	杨先生	13510535992	\N	50	健身路径	\N	杂牌	1	2	2009年	7	龙华街道港侨新村	[]	LH	1	三位扭腰器*2	正常	0
107	\N	\N	EPDM	富联二区4栋门前广场	霞姐	13714001670	\N	160	健身路径	\N	奥瑞特	1	11	2023年	12	富联二区4栋门前广场	[]	LH	1	告示牌，上肢牵引器，漫步机，多功能锻炼器，蹬力压腿训练器，骑马器，自重式下压训练器，三位扭腰器，太极揉推器，腰背按摩器，展背器	地面破损1平米	1
108	\N	\N	安全地垫	黄背坑公园山上	何欣庭	15014044094	\N	170平方	健身路径	\N	杂牌	1	150	2013年	13	黄背坑公园山上	[]	LH	2	上肢牵引器，伸背器，腹肌训练板，太极揉推器，象棋桌，双杠，蹬力器，腰背按摩器，太空漫步机，蹬力器，告示牌，骑马器，骑马器，三位扭腰器	正常	0
86	\N	\N	EPDM	龙华街道三联公园	杨先生	13510535992	\N	200	健身路径	\N	好家庭+奥瑞特+杂牌	1	20	2010年	7	龙华街道三联公园	[]	LH	1	告示牌，上肢牵引器，太极揉推器*3，三位扭腰器*2，腰背按摩器*3，压腿器，肋木架*2，二连单杠，腹肌板，双位扭腰器，双杠，蹬力器，展背器，四级压腿按摩器	蹬力器限位损坏，三位扭腰器转盘轴承损坏，腰背按摩器滚轮损坏	3
87	\N	\N	硅PU	龙华街道郭吓新村社区内	杨先生	13510535992	\N	500	篮球架	\N	/	1	1	2018年	7	龙华街道郭吓新村社区内	[]	LH	1	篮球场	正常	0
88	\N	\N	EPDM	龙华街道宝湖新村1	麦石好	13723752495	\N	50	健身路径	\N	杂牌	1	8	2015年	8	龙华街道宝湖新村1	[]	LH	1	蹬力器，腰背按摩器，单人自行车，太极揉推器，划船器，二连单杠，三位扭腰器，漫步机	单人自行车踏板缺失*1，腰背按摩器滚轮缺失	2
89	\N	\N	安全地垫	宝湖新村新碑村小广场	麦石好	13723752495	\N	60	儿童滑梯	\N	杂牌	1	1	2022-03-08	8	宝湖新村新碑村小广场	[]	LH	1	\N	正常	0
90	\N	\N	90平拼装+60平安全地垫	龙华街道力劲集团	麦石好	13723752495	\N	150	健身路径	\N	好家庭	1	15	2016年	8	龙华街道力劲集团	[]	LH	1	上肢牵引器，秋千，肋木架（杂牌3件）棋牌桌，大转轮，按摩揉推器，腰背按摩器，三位扭腰器，漫步机，太极揉推器，展背器，四级压腿按摩器，骑马器，腹肌板，告示牌，	棋牌桌凳子松动*3，骑马器限位损坏	2
91	\N	\N	硅PU	力劲集团	麦石好	13723752495	\N	500	篮球架	\N	杂牌	1	1	2019年	8	力劲集团	[]	LH	1	\N	无人使用	0
92	\N	\N	水泥地	龙华街道幸福城1期	麦石好	13723752495	\N	50	健身路径	\N	杂牌	1	6	2013年	8	龙华街道幸福城1期	[]	LH	1	双位腹肌板，三位扭腰器，漫步机，跷跷板，组合训练器，太极揉推器	正常	0
93	\N	\N	安全地垫	龙华街道幸福城3期	麦石好	13723752495	\N	170	健身路径	\N	好家庭	1	17	2014年	8	龙华街道幸福城3期	[]	LH	1	告示牌，腰背按摩器*2，上肢牵引器，二连单杠，漫步机*2，展背器，划船器，蹬力器，双位腹肌板，晃板扭腰器，太极揉推器，骑马器*2，四级压腿按摩器，三位扭腰器，	三位扭腰器转盘腐蚀*1，地面破损12平	2
94	\N	\N	拼装地板	龙华街道幸福城2期A	麦石好	13723752495	\N	50	健身路径	\N	奥瑞特	1	13	2013年	8	龙华街道幸福城2期A	[]	LH	1	太极揉推器*2，腰背按摩器，按摩揉推器，仰卧起坐训练器，三位上肢牵引器，三位扭腰器，斜躺健身车，告示牌，平步机，（其中三位上肢牵引器，三位扭腰器，太极揉推器，为杂牌）	正常	0
95	\N	\N	安全地垫	龙华街道盛世江南	麦石好	13723752495	\N	80	健身路径	\N	奥瑞特	1	13	2014年	8	龙华街道盛世江南	[]	LH	1	棋牌桌，三位扭腰器，平步机，斜躺健身车，仰卧起坐训练器，多功能训练器，蹬力器，告示牌，腰背按摩器，按摩揉推器，太极揉推器，漫步机，弹振压腿器	棋牌桌凳子腐蚀*1，三位扭腰器转盘损坏*1，平步机轴承损坏，太极揉推器转盘轴承损坏*1	4
96	\N	\N	硅PU	龙华街道高坳新村	陈欢	13824379178	\N	500	网球场	\N	杂牌	1	1	2011年	9	龙华街道高坳新村	[]	LH	1	网球场	正常	0
97	\N	\N	EPDM	龙华街道高坳新村	陈欢	13824379178	\N	100	健身路径	\N	杂牌	1	20	2008年	9	龙华街道高坳新村	[]	LH	1	双位腹肌板*2，平步机，太极揉推器*2，双位骑马器，单人骑马器，四级压腿按摩器，腰背按摩器，跷跷板*2，双杠*2，肋木架*2，漫步机，三位扭腰器，蹬力器，二连单杠，天梯	正常	0
98	\N	\N	EPDM	龙华街道玉翠A区1	陈欢	13824379178	\N	60	健身路径	\N	杂牌	1	8	2015年	9	龙华街道玉翠A区1	[]	LH	1	划船器，平步机，腹肌板*2，太极揉推器，漫步机，腿部按摩器，双杠	双杠盖帽丢失1，地面整体破旧	2
99	\N	\N	EPDM	龙华街道玉翠C区	陈欢	13824379178	\N	80	健身路径	\N	杂牌	1	8	2015年	9	龙华街道玉翠C区	[]	LH	1	健身车*2，双杠，漫步机，划船器，三位扭腰器，蹬力器，太极揉推器，	地面破损严重	1
100	\N	\N	EPDM	壹城中心1区5楼空中花园中心广场	周蓉蓉	19210053296	\N	60	健身路径	\N	奥瑞特	1	4	\N	10	壹城中心1区5楼空中花园中心广场	[]	LH	1	肋木架，天梯，三位扭腰器，漫步机。	三位扭腰器转盘损坏*2，地面破损2平米	2
101	\N	\N	拼装地板	龙华街道壹城中心九区	周蓉蓉	19210053296	\N	100	健身路径	\N	好家庭	1	10	2019年	10	龙华街道壹城中心九区	[]	LH	1	告示牌，上肢牵引器，太极揉推器*2，漫步机，上下肢训练器，骑马器，钟摆器，背肌训练器，压腿按摩器，	上下肢训练器限位损坏	1
102	\N	\N	拼装地板	壹城中心1区5楼空中花园中心广场	周蓉蓉	19210053296	\N	50	健身路径	\N	天行健	1	7	2019年	10	壹城中心1区5楼空中花园中心广场	[]	LH	1	扩胸训练器，拉力训练器，坐式蹬力训练器，腿部训练器，坐式前推训练器，推举训练器，告示牌	腿部训练器限位损坏	1
103	\N	\N	拼装地板	景新社区龙泽榕园	周蓉蓉	19210053296	\N	120	健身路径	\N	好家庭	1	12	2017年	10	景新社区龙泽榕园	[]	LH	1	上肢牵引器，肋木架，腰背按摩器，按摩揉推器，三位扭腰器，棋牌桌，骑马器，告示牌，腹肌板，四级压腿按摩器，太极揉推器，漫步机	正常	0
104	\N	\N	EPDM	龙华共和小区	李姐	13651467206	\N	80	健身路径	\N	杂牌未知	1	8	2017年	11	龙华共和小区	[]	LH	1	腹肌板，太极揉推器，三位扭腰器，平步机，单杠，椭圆机，蹬力器，单人做拉器	三位扭腰器转盘缺失3，整体器材老旧	2
105	\N	\N	安全地垫	龙华街道汇食街	李姐	13651467206	\N	120	健身路径	\N	杂牌未知	1	17	2012年	11	龙华街道汇食街	[]	LH	1	告示牌，腹肌板，钟摆器，坐立扭腰训练器，蹬力器，腰背按摩器*2，漫步机*3，太极揉推器，椭圆机*2，健身车，骑马器*2，扭腰步道	不属于管养，已计划明年整体更换	0
106	\N	\N	EPDM	龙华街道油园新村	李姐	13651467206	\N	300	健身路径	\N	好家庭+杂牌	1	29	2017年	11	龙华街道油园新村	[]	LH	1	三联单杠，三人蹬力器，四人蹬力器，上肢牵引器，天梯，棋牌桌*2，漫步机，太极揉推器，扭腰步道，告示牌，三位扭腰器，钟摆器，腿部按摩器{14件杂牌} 棋牌桌，双杠，太极揉推器，漫步机，骑马器，腹肌板，伸背器，告示牌，按摩揉推器，三位扭腰器，二连单杠，肋木架，上肢牵引器，展背器{15件好家庭}	漫步机轴承损坏*2上肢牵引器把手损坏，上肢牵引器轴承损坏*2，三位扭腰器转盘轴承损坏，告示牌补焊，多件器材补漆	6
261	\N	\N	硅PU	民治街道沙吓村篮球场大厦旁	舒畅	18998916326	\N	500平方	篮球场	\N	金陵	1	1	2008年	29	民治街道沙吓村篮球场大厦旁	[]	LH	3	篮球场	正常	0
109	\N	\N	硅pu	黄背坑公园东北门口	何欣庭	15014044094	\N	700平方	篮球场	\N	金陵	1	10	2013年	13	黄背坑公园东北门口	[]	LH	2	篮球场	篮板防撞条脱落，地面开裂，磨损严重	3
110	\N	\N	拼装地板	君子布社区公园	何欣庭	15014044094	\N	300平方	健身路径	\N	好家庭	1	170	2010年	13	君子布社区公园	[]	LH	2	象棋桌*4，肋木架，智能健身车，告示牌，腹背肌双功能训练器，深蹲提踵双功能训练器，竞赛自行车，上肢肩关节双功能训练器，告示牌，高拉推举双功能训练器，智能自发电肱二头肌训练器，推胸划船训练器，腿部屈伸双功能训练器，健身房	腿部屈伸双功能训练器右边手柄功能断缺，高位推举双功能训练器灯脱落，战绳盖帽缺失	3
111	\N	\N	EPDM	君子布老围新村1栋	何欣庭	15014044094	\N	100平方	健身路径	\N	好家庭	1	130	2014年	13	君子布老围新村1栋	[]	LH	2	肋木架，单杆，按摩揉推器，肋木架，双杠，上肢牵引器，告示牌，三位扭腰器，太空漫步机，腰背按摩器，伸背器，腹肌训练板，战绳	地面破损，太空漫步机盖帽缺失，象棋桌桌面面板脱落	3
112	\N	\N	EPDM	君子布老围新村126栋	何欣庭	15014044094	\N	90平方	健身路径	\N	澳瑞特	1	100	2013年	13	君子布老围新村126栋	[]	LH	2	双杠，单杠，告示牌，上肢牵引器，太极揉推器，体位前屈，腰背按摩器，三位扭腰器，弹振压腿器，太空漫步机	双杠脱漆，腰背按摩器两个滚轮损坏，地面破损	3
113	\N	\N	EPDM	君子布龙兴新村60号楼	何欣庭	15014044094	\N	150平方	健身路径	\N	澳瑞特	1	130	2013年	13	君子布龙兴新村60号楼	[]	LH	2	告示牌，太极揉推器，按摩揉推器，体位前屈，腰背按摩器，弹振压腿器，蹬力器，太空漫步机，单杠，肋木架，天梯，双杠，三位扭腰器	手部腿部按摩器滚轮盖帽缺失，腰背按摩器总成缺失，一个滚轮卡住另一个损坏，太空漫步机轴承损坏，三位扭腰器三转盘卡住	5
114	\N	\N	硅pu	龙兴新村60号楼	何欣庭	15014044094	\N	700平方	篮球场	\N	金陵	1	10	2013年	13	龙兴新村60号楼	[]	LH	2	篮球场	围网破损，地面磨损严重，篮板防撞条缺失，球网破损，球框缺失	5
115	\N	\N	\N	君子布新路	何欣庭	15014044094	\N	\N	健身路径	\N	\N	1	0	2013年	13	君子布新路	[]	LH	2	\N	\N	0
116	\N	\N	地砖	君子布张二新村7栋	何欣庭	15014044094	\N	70平方	健身路径	\N	好家庭	1	110	2013年	13	君子布张二新村7栋	[]	LH	2	告示牌，三位扭腰器，四级压腿器，步行器，腰背按摩器，腹肌训练板，蹬力器，太空漫步机，肋木架，太空漫步机，肋木架	二位蹬力器座椅立柱螺丝松动，三位扭腰器转盘卡住，太空漫步机防撞条缺失	3
117	\N	\N	\N	张一办公楼	何欣庭	15014044094	\N	\N	健身路径	\N	\N	1	0	2014年	13	张一办公楼	[]	LH	2	\N	\N	0
118	\N	\N	拼装地板	观禧花园三楼平台1栋	何欣庭	15014044094	\N	130平方	健身路径	\N	澳瑞特	1	110	2023年	13	观禧花园三楼平台1栋	[]	LH	2	告示牌，告示牌，腰背按摩器，太极揉推器，骑马器，上肢牵引器，多功能锻炼器，自重式下压训练器，太空漫步机，三位扭腰器，伸展器	太空漫步机防撞胶套缺失，	1
119	\N	\N	安全地垫	桂花新村九巷4号楼	陈巧茹	15999586300	\N	70平方	健身路径	\N	杂牌	1	50	2012年	14	桂花新村九巷4号楼	[]	LH	2	双杠，骑马器，蹬力器，太空漫步机，蹬力器	双杠盖帽缺失，太空漫步机立柱底部腐蚀报废，二位蹬力器立柱开裂，二位蹬力器防撞胶垫缺失，安全地垫老旧缺失	5
120	\N	\N	拼装地板	桂花新别墅区	陈巧茹	15999586300	\N	140平方	健身路径	\N	\N	1	0	2016年	14	桂花新别墅区	[]	LH	2	\N	\N	0
121	\N	\N	\N	赤花岭新村A栋	陈巧茹	15999586300	\N	\N	健身路径	\N	澳瑞特、红旗	1	160	2024年	14	赤花岭新村A栋	[]	LH	2	太极揉推器，步行器，四级压腿器，多功能按摩器，腰背按摩器，腹肌训练板，蹬力器，三位扭腰器，象棋桌，象棋桌，告示牌，骑马器，太空漫步机，步行器，伸背器，象棋桌	围棋桌坐凳缺失	1
122	\N	\N	丙烯酸	贵湖塘老围	陈巧茹	15999586300	\N	1400平方	健身路径	\N	\N	1	0	2016年	14	贵湖塘老围	[]	LH	2	\N	\N	0
123	\N	\N	丙烯酸	贵湖塘村	陈巧茹	15999586300	\N	1700平方	篮球场	\N	\N	1	0	2017年	14	贵湖塘村	[]	LH	2	\N	\N	0
124	\N	\N	\N	赤花岭新村A栋	陈巧茹	15999586300	\N	\N	篮球场	\N	金陵	1	20	2012年	14	赤花岭新村A栋	[]	LH	2	篮球场	地面磨损严重，篮球架立柱防撞套缺失，篮板防撞条缺失，篮筐缺失	4
125	\N	\N	\N	赤花岭新村桂花山庄A栋	陈巧茹	15999586300	\N	\N	网球场	\N	未知	1	20	2023年	14	赤花岭新村桂花山庄A栋	[]	LH	2	篮球场	正常	0
126	\N	\N	\N	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	陈巧茹	15999586300	\N	\N	中式台球桌	\N	未知	1	10	2022年	14	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	[]	LH	2	中式台球桌	球杆损坏一支	1
127	\N	\N	安全地垫	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	陈巧茹	15999586300	\N	90平方	健身路径	\N	酷威、鑫派	1	30	2022年	14	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	[]	LH	2	双杠，单杠，告示牌	双杠立柱底部脱漆	1
128	\N	\N	硅pu	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	陈巧茹	15999586300	\N	500平方	移动式篮球架	\N	未知	1	20	2022年	14	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	[]	LH	2	篮球场	正常	0
129	\N	\N	硅pu	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	陈巧茹	15999586300	\N	500平方	乒乓球台	\N	未知	1	20	2022年	14	龙华区人民武装部，龙华区观光路与惠民一路交叉口北220米	[]	LH	2	乒乓球台	正常	0
130	\N	\N	拼装地板	观澜硅谷动力数码产业园	黄国清	15725580009	\N	60	健身路径	\N	天行健	1	70	2022-10-31	15	观澜硅谷动力数码产业园	[]	LH	2	蹲举训练器，坐式蹬力训练器，腿部训练器，坐式前推训练器，推举训练器，扩胸训练器，告示牌	蹲举训练器限位损坏	1
131	\N	\N	/	观澜硅谷动力数码产业园	黄国清	15725580009	\N	10	轨道式 中国象棋	\N	未知	1	20	2022-10-31	15	观澜硅谷动力数码产业园	[]	LH	2	轨道象棋*2	象棋桌损坏1，象棋桌损坏2	2
132	\N	\N	/	观澜硅谷动力数码产业园	黄国清	15725580009	\N	/	篮球场	\N	未知	1	10	2022-10-31	15	观澜硅谷动力数码产业园	[]	LH	2	\N	正常	0
133	\N	\N	拼装地板	库坑新围新村9栋	杨金凤	13600431910	\N	125平方	健身路径	\N	舒华	1	120	2024年	16	库坑新围新村9栋	[]	LH	2	肋木架，单杆，椭圆机，腰背按摩器，骑马器，斜躺式健身车，蹬力器，划船器，太空漫步机，蹬力器，太极揉推器，三位扭腰器	正常	0
134	\N	\N	硅PU	库坑新围新村9栋	杨金凤	13600431910	\N	500平方	篮球场	\N	金陵	1	10	2016年	16	库坑新围新村9栋	[]	LH	2	篮球场	两个篮板蓝网损坏、防撞边条损坏	2
135	\N	\N	地砖	观澜街道库坑中心区24栋	杨金凤	13600431910	\N	100平方	健身路径	\N	好家庭	1	140	2014年	16	观澜街道库坑中心区24栋	[]	LH	2	肋木架，单杠，三位扭腰器，伸背器，腹肌板，告示牌，骑马器，太极推手器，腰背按摩器，双杠，太空漫步机 ，按摩揉推器，上肢牵引器	手部腿部按摩器转盘损坏，上肢牵引器轴承损坏	2
136	\N	\N	安全地垫	观澜街道库坑中心区长者服务中心	杨金凤	13600431910	\N	70平方	健身路径	\N	杂牌	1	80	2014年	16	观澜街道库坑中心区长者服务中心	[]	LH	2	椭圆机，椭圆机，蹬力器，骑马器，三位扭腰器，太空漫步机，腰背按摩器，太极揉推器	安全地垫缺失破损70平方，三位扭腰器转盘损坏三个	2
137	\N	\N	硅PU	观澜街道库坑中心区长者服务中心	杨金凤	13600431910	\N	500平方	篮球场	\N	金陵	1	10	2013年	16	观澜街道库坑中心区长者服务中心	[]	LH	2	篮球场	场地面漆磨损严重，两个蓝网损坏，一条防撞边条损坏	3
138	\N	\N	EPDM	观澜街道围仔小区21栋	杨金凤	13600431910	\N	130平方	健身路径	\N	杂牌、鑫派	1	60	2012年	16	观澜街道围仔小区21栋	[]	LH	2	扭腰器，三位扭腰器，椭圆机，大转轮，太极揉推器，腰背按摩器，	正常	0
139	\N	\N	硅PU	观澜街道围仔小区21栋	杨金凤	13600431910	\N	700平方	篮球场	\N	金陵	1	10	2012年	16	观澜街道围仔小区21栋	[]	LH	2	篮球场	地面起泡破损10平方	1
140	\N	\N	EPDM	观澜街道围仔新村5栋	杨金凤	13600431910	\N	180平方	健身路径	\N	杂牌	1	80	2013年	16	观澜街道围仔新村5栋	[]	LH	2	三位扭腰器，手部揉推器，手部揉推器，太极揉推器，滚轮，告示牌，多功能训练器，大转轮	地面开裂严重	1
141	\N	\N	草地	观澜街道宝三和工业区西南门	杨金凤	13600431910	\N	120平方	健身路径	\N	杂牌	1	100	2010年	16	观澜街道宝三和工业区西南门	[]	LH	2	太空漫步机，蹬力器，双人双杠，钟摆器，椭圆机，腰背按摩器，腹肌板，四级压腿器，三位扭腰器，坐拉训练器	太空漫步机轴承损坏，蹬力器盖帽缺失，双杠扶手脱焊，平步机腐蚀断缺报废，	4
142	\N	\N	草地	观澜街道国桥工业园二期东北门	杨金凤	13600431910	\N	130平方	健身路径	\N	澳瑞特	1	150	2014年	16	观澜街道国桥工业园二期东北门	[]	LH	2	告示牌，蹬力器，单杠，弹振压腿器，腰背按摩器，太极揉推器，三位扭腰器，伸背器，椭圆机，斜躺式自行车，多功能训练器，仰卧起坐板，象棋桌，按摩揉推器，太空漫步机	正常	0
143	\N	\N	硅PU	观澜街道国桥二期东北门	杨金凤	13600431910	\N	450平方	篮球场	\N	好家庭	1	10	2010年	16	观澜街道国桥二期东北门	[]	LH	2	篮球场	地面开裂，面漆磨损20平方	2
144	\N	\N	EPDM	观澜街道库坑社区公园前门	杨金凤	13600431910	\N	45平方	健身路径	\N	杂牌	1	60	2016年9月	16	观澜街道库坑社区公园前门	[]	LH	2	椭圆机，三位扭腰器，太空漫步机，太极揉推器，钟摆器，腿部按摩器	正常	0
145	\N	\N	拼装地板	硅谷动力工业园（硅谷动力射频科技园宿舍楼后面）	杨金凤	13600431910	\N	35平方	健身路径	\N	天行健	1	60	2018-01-01	16	硅谷动力工业园（硅谷动力射频科技园宿舍楼后面）	[]	LH	2	告示牌，腿部训练器，坐式蹬力训练器，蹲举训练器，扩胸训练器，推举训练器，坐式前推训练器	腿部训练器限位损坏，坐式蹬力器限位损坏，蹲举训练器限位损坏，扩胸训练器限位损坏，推举训练器限位损坏，坐式前推训练器限位损坏	6
146	\N	\N	水磨石	硅谷动力工业园（硅谷动力射频科技园宿舍楼后面）	杨金凤	13600431910	\N	420平方	篮球场	\N	金陵	1	10	2018-01-01	16	硅谷动力工业园（硅谷动力射频科技园宿舍楼后面）	[]	LH	2	篮球场	正常	0
147	\N	\N	EPDM	观澜街道桂花公园（桂香公园）	罗媛媛	21031607	\N	100平方	健身路径	\N	好家庭	1	80	2010年	18	观澜街道桂花公园（桂香公园）	[]	LH	2	肋木架，上肢牵引器，腹肌板，腰背按摩器，三位扭腰器，太空漫步机，双人双杠，四级压腿器，	EPDM面层破损50平方，四级压腿器滚轮损坏	2
148	\N	\N	硅pu	观澜街道桂花社区公园（桂香公园）	罗媛媛	21031607	\N	500平方	篮球场	\N	金陵	1	10	2012年	18	观澜街道桂花社区公园（桂香公园）	[]	LH	2	篮球场	两篮网损坏、两防撞边条缺失、一篮球架支架支撑固定螺丝缺失，地面层磨损	4
149	\N	\N	拼装地板	爱心家园3栋	罗媛媛	21031607	\N	60平方	儿童滑梯	\N	好家庭	1	10	2022-03-08	18	爱心家园3栋	[]	LH	2	儿童乐园	正常	0
150	\N	\N	拼装地板	爱心家园1栋	罗媛媛	21031607	\N	115平方	健身路径	\N	澳瑞特	1	120	2024年	18	爱心家园1栋	[]	LH	2	单杠，多功能训练器，上肢牵引器，三位扭腰器，太空漫步机，蹬力压腿训练器，太极揉推器，腰背按摩器，伸背器，蹬力器，自重式下压训练器，告示牌	正常	0
151	\N	\N	拼装地板	观澜街道蚌岭新村5栋	杨金凤	13600431910	\N	45平方	健身路径	\N	澳瑞特	1	110	2012年	18	观澜街道蚌岭新村5栋	[]	LH	2	蹬力器，自重式下压训练器，太空漫步机，三位扭腰器，伸背器，手转器，腰背按摩器，太极揉推器，多功能训练器，上肢牵引器，告示牌	正常	0
152	\N	\N	拼装地板	观澜街道大沙河新村27栋	杨金凤	13600431910	\N	100平方	健身路径	\N	澳瑞特	1	110	2024年	18	观澜街道大沙河新村27栋	[]	LH	2	告示牌，太空漫步机，三位扭腰器，自重式下压训练器，蹬力器，手转器，腰背按摩器，伸背器，太极揉推器，上肢牵引器，多功能训练器	正常	0
153	\N	\N	安全地垫	观澜街道大沙河新村28栋	杨金凤	13600431910	\N	35平方	健身路径	\N	杂牌	1	40	2008年	18	观澜街道大沙河新村28栋	[]	LH	2	三位扭腰器，腹肌板，太空漫步机，双人双杠	三位扭腰器两个转盘损坏，安全地垫老旧	2
154	\N	\N	拼装地板	观澜街道牛湖老二村48栋	汪昆	13689503663	\N	\N	健身路径	\N	澳瑞特	1	120	2017年7月	19	观澜街道牛湖老二村48栋	[]	LH	2	扭腰步道，单杠，三位扭腰器，太空漫步机，太极揉推器，按摩揉推器，伸背器，仰卧起坐板，蹬力器，椭圆机，腰背按摩器，象棋桌，	健骑机立柱松动	1
155	\N	\N	硅pu	观澜街道牛湖老二村48栋	汪昆	13689503663	\N	600平方	篮球场	\N	杂牌	1	10	2013年	19	观澜街道牛湖老二村48栋	[]	LH	2	篮球场	一个篮网破损，篮球架底座盖板螺丝缺失	2
156	\N	\N	地砖	观澜街道牛湖社区工作站后面公园	汪昆	13689503663	\N	140平方	健身路径	\N	舒华	1	120	2022年	19	观澜街道牛湖社区工作站后面公园	[]	LH	2	单杠，肋木架，天梯，双杠，腹肌板，告示牌，上肢牵引器，腰背按摩器，太空漫步机，太极揉推器，三位扭腰器，象棋桌	三位扭腰器转盘卡住	1
157	\N	\N	拼装地板	观澜街道石一新村船岭16栋旁	汪昆	13689503663	\N	100平方	健身路径	\N	澳瑞特	1	130	2020年	19	观澜街道石一新村船岭16栋旁	[]	LH	2	太极揉推器，象棋桌，腰背按摩器，按摩揉推器，仰卧起坐板，斜躺式自行车，骑马器，蹬力器，伸背器，三位扭腰器，肋木架，上肢牵引器，告示牌	腰背按摩器滚轮损坏，伸展器限位损坏	2
158	\N	\N	丙烯酸	观澜街道石一新村船岭16栋旁	汪昆	13689503663	\N	650平方	篮球场	\N	金陵	1	10	2020年	19	观澜街道石一新村船岭16栋旁	[]	LH	2	篮球场	地面磨损70平方	1
159	\N	\N	拼装地板	观澜街道牛湖社区石二村小区10栋	汪昆	13689503663	\N	155平方	健身路径	\N	澳瑞特	1	130	2020年	19	观澜街道牛湖社区石二村小区10栋	[]	LH	2	太空漫步机，象棋桌，椭圆机，按摩揉推器，太极揉推器，伸背器，腰背按摩器，仰卧起坐板，蹬力器，单杠，告示牌	象棋桌面烧焦，三位扭腰器转盘缺失一个，健骑机限位损坏	3
160	\N	\N	拼装地板	观澜街道牛湖老一村篮球场旁	汪昆	13689503663	\N	40平方	健身路径	\N	澳瑞特	1	60	2020年	19	观澜街道牛湖老一村篮球场旁	[]	LH	2	仰卧起坐板，单杠，太空漫步机，蹬力器，按摩揉推器，三位扭腰器，	健骑机限位损坏	1
161	\N	\N	安全地垫＋地砖	观澜街道牛湖求雨岭城市公园	汪昆	13689503663	\N	170平方	健身路径	\N	舒华	1	150	2022年	19	观澜街道牛湖求雨岭城市公园	[]	LH	2	上肢牵引器，太极揉推器，太空漫步机，腰背按摩器，双杠，腹肌板，天梯，告示牌，双杠，单杠，单杠，三位扭腰器，太极揉推器，象棋桌，腰背按摩器	正常	0
162	\N	\N	拼装地板	牛湖社区求雨岭城市公园	汪昆	13689503663	\N	130平方	室外健身路径—塑木系列室外健身器材	\N	好家庭、杂牌	1	130	2020-12-21	19	牛湖社区求雨岭城市公园	[]	LH	2	太极推手器，蹬力器，骑马器，上肢牵引器，肋木架，四级压腿器，钟摆器，伸展器，太空漫步机，钟摆器，骑马器，腰背按摩器，立式健身车，	晃板扭腰器立柱倾斜，健骑机限位损坏，伸展器限位损坏，太空漫步机立柱下陷导致摆腿高低不平，健骑机立柱松动	5
163	\N	\N	丙烯酸	求雨岭城市公园	汪昆	13689503663	\N	400平方	羽毛球场	\N	不详	1	10	2022-11-29	19	求雨岭城市公园	[]	LH	2	羽毛球场	三张球网破损，地面破损10平方	2
164	\N	\N	丙烯酸	求雨岭城市公园	汪昆	13689503663	\N	650平方	改建羽毛球场（网球场，计划改羽毛球场）	\N	不详	1	10	2020年	19	求雨岭城市公园	[]	LH	2	羽毛球场	球网破损，场地边上地面有塌陷，开裂严重	3
165	\N	\N	EPDM	观澜街道牛湖石三小区1栋	汪昆	13689503663	\N	155平方	健身路径	\N	澳瑞特	1	130	2020-01-01	19	观澜街道牛湖石三小区1栋	[]	LH	2	单杠，伸背器，腰背按摩器，椭圆机，三位扭腰器，单杠，按摩揉推器，太空漫步机，扭腰步道，象棋桌，斜躺式健身车，仰卧起坐板，太极揉推器	象棋桌桌面变形损坏，告示牌牌面缺失	2
166	\N	\N	拼装地板+安全地垫	观澜街道大水田社区工作站	叶小姐	13530466708	\N	110平方	健身路径	\N	好家庭	1	90	2024年	20	观澜街道大水田社区工作站	[]	LH	2	深蹲提踵双功能训练器，健身房，拉伸机，高拉推举双功能训练器，腿部屈伸双功能训练器，上肢肩关节双功能训练器，推胸划船训练器，智能健身车	健骑机立柱松动	1
167	\N	\N	拼装地板	观澜街道大水田西区36栋旁	叶小姐	13530466708	\N	55平方	健身路径	\N	杂牌	1	40	2008年	20	观澜街道大水田西区36栋旁	[]	LH	2	太空漫步机，跷跷板，椭圆机，太极推手器	腰背按摩器滚轮损坏，伸展器限位损坏	2
168	\N	\N	安全地垫	观澜街道大水田西区物业管理处旁	叶小姐	13530466708	\N	110平方	健身路径	\N	好家庭	1	80	2008年	20	观澜街道大水田西区物业管理处旁	[]	LH	2	骑马器，三位扭腰器，太空漫步机，太空漫步机，腰背按摩器，蹬力器，上肢牵引器，告示牌	三位扭腰器转盘卡住	1
169	\N	\N	硅pu	观澜街道大水田街心公园	叶小姐	13530466708	\N	580平方	篮球场	\N	金陵	1	10	2021年	20	观澜街道大水田街心公园	[]	LH	2	篮球场	一个篮网破损，篮球架底座盖板螺丝缺失	2
170	\N	\N	硅pu	观澜街道大水田工作站	叶小姐	13530466708	\N	600平方	篮球场	\N	杂牌	1	10	2012年	20	观澜街道大水田工作站	[]	LH	2	篮球场	地面磨损70平方	1
171	\N	\N	拼装地板	观澜版画基地	叶小姐	13530466708	\N	80平方	室外健身路径—塑木系列室外健身器材	\N	好家庭	1	120	2020-12-21	20	观澜版画基地	[]	LH	2	告示牌，太极推手器，腰背按摩器，钟摆器，四级压腿器，上肢牵引器，肋木架，蹬力器，骑马器，蹬力器，拉伸机，太空漫步机，	正常	0
172	\N	\N	拼装地板	版画基地	叶小姐	13530466708	\N	550平方	篮球场	\N	好家庭	1	10	2020-12-21	20	版画基地	[]	LH	2	篮球场	健骑机限位损坏	1
173	\N	\N	丙烯酸	观澜街道大富路20号硅谷动力智能终端产业园B3栋	陈先生	13392194602	\N	420平方	篮球场	\N	不详	1	10	2020年	21	观澜街道大富路20号硅谷动力智能终端产业园B3栋	[]	LH	2	篮球场	地面开裂	1
174	\N	\N	拼装地板	观澜街道大富路20号硅谷动力智能终端产业园B2栋	陈先生	13392194602	\N	80平方	健身路径	\N	好家庭	1	120	2020年	21	观澜街道大富路20号硅谷动力智能终端产业园B2栋	[]	LH	2	告示牌，象棋桌，太空漫步机，伸背器，三位扭腰器，四级压腿器，太极推手器，腰背按摩器，骑马器，按摩揉推器，大转轮，腹肌板	正常	0
175	\N	\N	地砖	丰盛市场（大富桂月公园）	陈先生	13392194602	\N	120平方	健身路径	\N	杂牌	1	70	2022年	21	丰盛市场（大富桂月公园）	[]	LH	2	腰背按摩器，腹肌板，三位扭腰器，蹬力器，太极推手器，太极揉推器，伸背器	正常	0
176	\N	\N	水磨石	汽车电子创业园	陈先生	13392194602	\N	900平方	篮球场	\N	不详	1	10	\N	21	汽车电子创业园	[]	LH	2	篮球场	正常	0
305	\N	\N	\N	民治街道上河坊1-B座	徐观平	13534171844	\N	\N	健身路径	\N	\N	1	0	2014年	33	民治街道上河坊1-B座	[]	LH	3	\N	\N	0
220	\N	\N	EPDM	民治街道水榭春天123期11栋	段小姐	18569665058	\N	80平方	健身路径	\N	杂牌	1	7	2015年	26	民治街道水榭春天123期11栋	[]	LH	3	告示牌，呼啦圈，椭圆机，漫步机，蹬力器，三位扭腰器，上肢牵引器，	平步机轴承损坏	1
177	\N	\N	拼装地板	老街（观澜大道385-389号门口）	陈业琪	13751110305	\N	90平方	健身路径	\N	好家庭	1	140	2016年	22	老街（观澜大道385-389号门口）	[]	LH	2	太极推手器，腰背按摩器，肋木架，上肢牵引器，按摩揉推器，四级压腿器，蹬力器，骑马器，三位扭腰器，腹肌板，太空漫步机，伸背器，告示牌，象棋桌	拼装地板缺失15平方，手部腿部按摩器转盘损坏，腰背按摩器总成固定螺丝缺失，象棋桌座椅损坏两个	4
178	\N	\N	水泥地	观澜街道大布巷统建楼1栋	陈业琪	13751110305	\N	110平方	健身路径	\N	英派斯	1	110	2016年	22	观澜街道大布巷统建楼1栋	[]	LH	2	腰背按摩器，椭圆机，拉伸机，太极揉推器，蹬力器，按摩揉推器，立式健身车，三位扭腰器，腹肌板，三位扭腰器，告示牌	椭圆机边条、盖帽缺失；伸展器边条、盖帽缺失；健身车脱漆，告示牌报废	6
179	\N	\N	EPDM	观澜街道贵苑花园B栋	陈业琪	13751110305	\N	95平方	健身路径	\N	爱之美	1	70	2008年	22	观澜街道贵苑花园B栋	[]	LH	2	太空漫步机，蹬力器，三位扭腰器，划船器，告示牌，腹肌板，单杠	告示牌牌面缺失	1
180	\N	\N	/	观澜湖新城商场	-	21059466	\N	/	健身路径	\N	不详	1	10	2016年	23	观澜湖新城商场	[]	LH	2	篮球架	一个篮网损坏	1
181	\N	\N	拼装地板	老年活动中心	-	21059466	\N	80平方	健身路径	\N	杂牌	1	110	2016年	23	老年活动中心	[]	LH	2	告示牌，太空漫步机，单杠，仰卧起坐板，腰背按摩器，钟摆器，骑马器，椭圆机，太极揉推器，按摩揉推器，三位扭腰器，上肢牵引器，蹬力器，臂力训练器	场地已荒废，杂草丛生，无人使用	0
182	\N	\N	拼装地板	民治街道绿景香颂花园D栋	詹先生	13632731334	\N	40平方	健身路径	\N	杂牌	1	6	2011年	24	民治街道绿景香颂花园D栋	[]	LH	3	单杠，太空漫步机，上肢牵引器，跷跷板，腰背按摩器，臂力训练器	跷跷板轴承松动，太空漫步机轴承损坏	2
183	\N	\N	地砖	民治街道金地梅陇镇花园一期1栋	詹先生	13632731334	\N	20平方	健身路径	\N	澳瑞特	1	2	2013年	24	民治街道金地梅陇镇花园一期1栋	[]	LH	3	双杠，肋木架	正常	0
184	\N	\N	拼装地板	民治街道金地梅陇镇花园二期10栋	詹先生	13632731334	\N	80平方	健身路径	\N	澳瑞特、体之杰	1	12	2013年	24	民治街道金地梅陇镇花园二期10栋	[]	LH	3	蹬力器，太极推手器，柔韧训练器，滚桶，滚桶，太空漫步机，告示牌，腰背按摩器，三位扭腰器，按摩揉推器，蹬力器，上肢牵引器	太极揉推器转盘总成缺失一边	1
185	\N	\N	拼装地板	民治街道金地梅陇镇花园二期12栋	詹先生	13632731334	\N	140平方	健身路径	\N	澳瑞特、体之杰	1	15	2013年	24	民治街道金地梅陇镇花园二期12栋	[]	LH	3	太极推手器，钟摆器，肋木架，太空漫步机，腰背按摩器，坐拉器，椭圆机，坐拉器，双杠，太极推手器，腰背按摩器，三人蹬力器，告示牌，椭圆机，多功能训练器	钟摆器轴承损坏，太空漫步机轴承损坏	0
186	\N	\N	拼装地板	金地梅陇镇花园一期	詹先生	13632731334	\N	180平方	篮球场	\N	金陵	1	1	2011年	24	金地梅陇镇花园一期	[]	LH	3	篮球场	正常	0
187	\N	\N	拼装地板	民治街道阳光新境园G栋	詹先生	13632731334	\N	120平方	健身路径	\N	好家庭	1	14	2016年	24	民治街道阳光新境园G栋	[]	LH	3	三位扭腰器，骑马器，太极推手器，按摩揉推器，腹肌板，腰背按摩器，四级压腿器，伸背器，告示牌，上肢牵引器，蹬力器，肋木架，太空漫步机，棋牌桌	正常	0
188	\N	\N	拼装地板	民治街道风和日丽3栋	詹先生	13632731334	\N	110平方	健身路径	\N	好家庭	1	12	2013年	24	民治街道风和日丽3栋	[]	LH	3	肋木架，单杠，告示牌，双杠，太空漫步机，太空漫步机，腰背按摩器，三位扭腰器，骑马器，太极揉推器，四级压腿器，太极推手器，	太极揉推器手柄缺失，健骑机轴承缺失损坏，四级压腿器滚轮缺失两个	3
189	\N	\N	硅PU	风和日丽小区34栋	詹先生	13632731334	\N	500平方	篮球场	\N	宏康	1	1	2023年	24	风和日丽小区34栋	[]	LH	3	篮球场	正常	0
190	\N	\N	硅PU	风和日丽小区34栋	詹先生	13632731334	\N	500平方	羽毛球场	\N	不详	1	3	2023年	24	风和日丽小区34栋	[]	LH	3	羽毛球场	球网缺失三套	1
191	\N	\N	EPDM	新华城b栋平台	詹先生	13632731334	\N	120平方	健身路径	\N	澳瑞特	1	11	2023年	24	新华城b栋平台	[]	LH	3	告示牌，上肢牵引器，多功能训练器，太空漫步机，腰背按摩器，太极揉推器，三位扭腰器，伸背器，自重式下压训练器，骑马器，蹬力压腿训练器	正常	0
192	\N	\N	EPDM	民治街道嘉龙山庄羽毛球场120栋	詹先生	13632731334	\N	130平方	羽毛球场	\N	不详	1	1	2017年	24	民治街道嘉龙山庄羽毛球场120栋	[]	LH	3	羽毛球场	地面破损2平方，球网缺失	2
193	\N	\N	拼装地板	新牛工作站嘉龙山庄嘉华园健身广场120栋	詹先生	13632731334	\N	300平方	健身路径	\N	体之杰、好家庭、澳瑞特	1	32	2017年、2013年	24	新牛工作站嘉龙山庄嘉华园健身广场120栋	[]	LH	3	双杠，太空漫步机，单杠，多功能训练器，三位扭腰器，上肢牵引器，肋木架，呼啦桥，太空漫步机，弹振压腿器，蹬力器，四级压腿器，跷跷板，肋木架，棋牌桌，告示牌，腹肌板，太空漫步机，三位扭腰器，太极推手器，伸背器，太极推手器，按摩揉推器，按摩揉推器，腰背按摩器，骑马器，上肢牵引器，腰背按摩器，太极揉推器，上肢牵引器，告示牌，乒乓球台	腰背按摩器扶手变形，太极揉推器轴承损坏，腰背伸展器立柱腐蚀	3
194	\N	\N	草地	民治街道嘉龙山庄63栋	詹先生	13632731334	\N	120平方	健身路径	\N	好家庭	1	14	2016年	24	民治街道嘉龙山庄63栋	[]	LH	3	棋牌桌，告示牌，腹肌板，太空漫步机，伸背器，太极推手器，骑马器，三位扭腰器，蹬力器，四级压腿器，腰背按摩器，按摩揉推器，上肢牵引器，肋木架	正常	0
195	\N	\N	丙烯酸	民治街道嘉龙山庄篮球场120栋	詹先生	13632731334	\N	500平方	篮球场	\N	金陵	1	1	2017年	24	民治街道嘉龙山庄篮球场120栋	[]	LH	3	篮球场	地面开裂破损	1
196	\N	\N	水磨石地面	民治街道牛栏前村篮球场	詹先生	13632731334	\N	1000平方	篮球场	\N	金陵	1	2	2017年	24	民治街道牛栏前村篮球场	[]	LH	3	篮球场	三个篮板防撞胶条缺失，篮网破损缺失两个	2
197	\N	\N	拼装地板	民治街道苹果园2栋	詹先生	13632731334	\N	120平方	健身路径	\N	澳瑞特	1	14	2013年	24	民治街道苹果园2栋	[]	LH	3	多功能训练器，斜躺式健身车，三位扭腰器，象棋桌，告示牌，蹬力器，按摩揉推器，腰背按摩器，太极揉推器，伸背器，仰卧起坐练习器，太空漫步机，弹振压腿器，单杠	正常	0
198	\N	\N	EPDM	民治街道锦锈江南一期c2栋旁	詹先生	13632731334	\N	90平方	健身路径	\N	好家庭、澳瑞特	1	10	2013年	24	民治街道锦锈江南一期c2栋旁	[]	LH	3	天梯，肋木架，三位扭腰器，扭腰步道，太极推手器，四位蹬力器，告示牌，斜躺式健身车，单杠，弹振压腿器，	太极揉推器手柄缺失一个，四位蹬力器坐凳缺失一个、背靠损坏一个；	3
199	\N	\N	地砖、EPDM	民治街道锦锈江南二期7栋	詹先生	13632731334	\N	60平方	健身路径	\N	澳瑞特、体之杰	1	8	2015年	24	民治街道锦锈江南二期7栋	[]	LH	3	腰背按摩器，伸背器，太极推手器，腹肌板，伸背器，太极揉推器，步行器，腰背按摩器，	正常	0
200	\N	\N	丙烯酸	民治街道锦锈江南二期5栋篮球场	詹先生	13632731334	\N	470平方	篮球场	\N	金陵	1	1	2010年	24	民治街道锦锈江南二期5栋篮球场	[]	LH	3	篮球场	两个篮板防撞胶条缺失，	1
201	\N	\N	拼装地板	民治街道锦锈江南四期10栋	詹先生	13632731334	\N	140平方	健身路径	\N	好家庭、澳瑞特	1	14	2019年	24	民治街道锦锈江南四期10栋	[]	LH	3	告示牌，太极推手器，按摩揉推器，天梯，腰背按摩器，太空漫步机，三位扭腰器，太极推手器，四级压腿器，伸背器，骑马器，腹肌板，棋牌桌，仰卧起坐训练器，	太空漫步机缺失两个盖帽，健骑机限位损坏	2
202	\N	\N	拼装地板	华美丽苑太阳广场休闲区6栋	詹先生	13632731334	\N	70平方	健身路径	\N	澳瑞特、杂牌	1	12	2023年	24	华美丽苑太阳广场休闲区6栋	[]	LH	3	太极揉推器，伸背器，腰背按摩器，上肢牵引器，三位扭腰器，太空漫步机，太空漫步机，太空漫步机，多功能训练器，自重式下压训练器，骑马器，蹬力压腿训练器	正常	0
203	\N	\N	丙烯酸	华美丽苑小区2栋	詹先生	13632731334	\N	240平方	篮球场	\N	澳瑞特	1	1	2019年	24	华美丽苑小区2栋	[]	LH	3	篮球场	地面开裂；两个篮板防撞胶条缺失，	2
204	\N	\N	EPDM	民治街道金亨利首府E座	俊逸	18278657087	\N	40平方	健身路径	\N	杂牌	1	5	2010年	25	民治街道金亨利首府E座	[]	LH	3	腰背按摩器，太极推手器，太空漫步机，三位扭腰器，单杠，	太极揉推器手柄缺失	1
205	\N	\N	丙烯酸	民治街道创业花园篮球场109栋	俊逸	18278657087	\N	200平方	篮球场	\N	不详	1	1	2013年	25	民治街道创业花园篮球场109栋	[]	LH	3	篮球场	正常	0
206	\N	\N	地砖	民治街道莱蒙水榭山10-3栋	俊逸	18278657087	\N	20平方	健身路径	\N	杂牌	1	4	2014年	25	民治街道莱蒙水榭山10-3栋	[]	LH	3	坐推训练器，太极推手器，太空漫步机，腰背按摩器，	正常	0
207	\N	\N	地砖	民治街道莱蒙水榭山3-1栋	俊逸	18278657087	\N	21平方	健身路径	\N	杂牌	1	3	2015年	25	民治街道莱蒙水榭山3-1栋	[]	LH	3	划船器，压腿器，腰背按摩器，	正常	0
208	\N	\N	拼装地板	民治街道龙悦居四期9栋	俊逸	18278657087	\N	80平方	健身路径	\N	澳瑞特	1	11	2024年	25	民治街道龙悦居四期9栋	[]	LH	3	上肢牵引器，多功能训练器，太空漫步机，太极揉推器，三位扭腰器，伸背器，骑马器，自重式下压训练器，转手器，腰背按摩器，告示牌	正常	0
209	\N	\N	丙烯酸	民治街道龙悦居四期8栋	俊逸	18278657087	\N	450平方	篮球场	\N	不详	1	1	2009年	25	民治街道龙悦居四期8栋	[]	LH	3	篮球场	篮板防撞胶条缺失，篮网缺失，地面破损20平方	3
210	\N	\N	拼装地板	民治街道龙悦居一期C栋	俊逸	18278657087	\N	70平方	健身路径	\N	澳瑞特	1	11	2024年	25	民治街道龙悦居一期C栋	[]	LH	3	上肢牵引器，多功能训练器，太空漫步机，太极揉推器，三位扭腰器，伸背器，骑马器，自重式下压训练器，转手器，腰背按摩器，告示牌	正常	0
211	\N	\N	拼装地板	民治街道龙悦居二期4栋	俊逸	18278657087	\N	80平方	健身路径	\N	澳瑞特	1	11	2024年	25	民治街道龙悦居二期4栋	[]	LH	3	上肢牵引器，多功能训练器，太空漫步机，太极揉推器，三位扭腰器，伸背器，骑马器，自重式下压训练器，转手器，腰背按摩器，告示牌	自重式下压训练器轴承固定螺丝缺失	1
212	\N	\N	地砖	民治街道龙悦居二期6栋	俊逸	18278657087	\N	6平方	健身路径	\N	杂牌	1	1	2014年	25	民治街道龙悦居二期6栋	[]	LH	3	太极推手器	正常	0
213	\N	\N	地砖、拼装地板	民治街道龙悦居三期2栋	俊逸	18278657087	\N	170平方	健身路径	\N	好家庭/澳瑞特	1	20	2010年/2024年	25	民治街道龙悦居三期2栋	[]	LH	3	告示牌，太极推手器，单杠，三位扭腰器，骑马器，肋木架，腰背按摩器，双杠，太空漫步机，自重式下压训练器，告示牌，骑马器，伸背器，太极揉推器，转手器，三位扭腰器，太空漫步机，腰背按摩器，多功能训练器，上肢牵引器，	两件健骑机限位损坏，太空漫步机摆腿缺失，自重式下压训练器限位损坏，	3
214	\N	\N	EPDM	民治街道圣莫丽斯	俊逸	18278657087	\N	80平方	健身路径	\N	杂牌	1	15	2024年	25	民治街道圣莫丽斯	[]	LH	3	天梯，单杠，三位扭腰器，蹬力器，太极推手器，太空漫步机，上肢牵引器，腰背按摩器，太空漫步机，转轮，步行器，伸背器，骑马器，腹肌板，按摩揉推器，	正常	0
215	\N	\N	EPDM	北站中心公园	俊逸	18278657087	\N	350平方	智能健身房儿童乐园内	\N	好家庭	1	10	2020年	25	北站中心公园	[]	LH	3	告示牌，健身房，上肢·肩关节双功能训练器，立式健身车，深蹲·提纵双功能训练器，高拉·推举双功能训练器，腹背肌训练器，智能竞赛车，组合训练器B，战绳	深蹲提纵双功能训练器左右两边肩护垫损坏，组合训练器脱漆	2
216	/	\N	/	民治街道圣莫丽斯1篮球场	俊逸	18278657087	\N	/	篮球场	\N	/	1	0	2010年	25	民治街道圣莫丽斯1篮球场	[]	LH	3	私人球场	/	0
217	\N	\N	丙烯酸	民治街道德逸公园	段小姐	18569665058	\N	1000	篮球场	\N	不详	1	2	2013年	26	民治街道德逸公园	[]	LH	3	篮球场	四个篮板防撞胶条缺失，篮网缺失四个，地面磨损开裂	3
218	\N	\N	拼装地板	民治街道德逸公园	段小姐	18569665058	\N	360平方	健身路径	\N	澳瑞特	1	24	2012年	26	民治街道德逸公园	[]	LH	3	棋牌桌，腹肌板，告示牌，腹肌板，蹬力压腿训练器，太极推手器，划船器，蹬力器，蹬力器，按摩揉推器，腰背按摩器，按摩揉推器，太空漫步机，太空漫步机，大转轮，多功能按摩器，蹬力器，蹬力器，前举训练器，单杠，伸背器，大转轮，上肢牵引器，上肢牵引器	大转轮手柄缺失	1
219	\N	\N	EPDM	民治街道水榭春天123期10栋	段小姐	18569665058	\N	30平方	健身路径	\N	杂牌	1	3	2015年	26	民治街道水榭春天123期10栋	[]	LH	3	肋木架，单杆，双杠	正常	0
221	\N	\N	拼装地板	民治街道城投七里香榭	段小姐	18569665058	\N	70平方	健身路径	\N	好家庭	1	11	2016年	26	民治街道城投七里香榭	[]	LH	3	伸背器，太极推手器，四级压腿器，腰背按摩器，按摩揉推器，蹬力器，告示牌，肋木架，骑马器，棋牌桌，腹肌板，	二位蹬力器两边限位损坏，健骑机限位损坏，	2
222	\N	\N	地砖	民治街道日岀印象A区5栋	段小姐	18569665058	\N	40平方	健身路径	\N	贝尔康	1	3	2012年	26	民治街道日岀印象A区5栋	[]	LH	3	三位扭腰器，太极推手器，蹬力器	太极揉推器转盘损坏两个，蹬力器桌凳缺失一边	2
223	\N	\N	地砖	民治街道日岀印象A区6栋	段小姐	18569665058	\N	30平方	健身路径	\N	贝尔康、澳瑞特、体之杰	1	6	2008年	26	民治街道日岀印象A区6栋	[]	LH	3	椭圆机，肩关节康复器，伸展器，腹肌板，太空漫步机，告示牌	双人联动漫步机轴承损坏，太极揉推器手柄缺失，伸展器扶手损坏缺失一边	3
224	\N	\N	拼装地板	民治街道中央原著御珑苑1栋	段小姐	18569665058	\N	80平方	健身路径	\N	好家庭	1	14	2019年	26	民治街道中央原著御珑苑1栋	[]	LH	3	腹肌板，太空漫步机，太极推手器，三位扭腰器，四级压腿器，按摩揉推器，腰背按摩器，肋木架，蹬力器，上肢牵引器，骑马器，伸背器，棋牌桌，告示牌，	按摩揉推器转盘损坏、二位蹬力器限位损坏，健骑机限位损坏	3
225	\N	\N	拼装地板	民治街道中央原著藏珑苑4栋	段小姐	18569665058	\N	70平方	健身路径	\N	澳瑞特	1	11	2019年	26	民治街道中央原著藏珑苑4栋	[]	LH	3	裸关节屈伸练习器，环形上肢协调功能练习器，肩梯/上肢协调功能练习器，背部伸展/腕关节/提力练习器，肩关节回旋/前臂、腕关节练习器，平行杠步态练习器，骑马器，太极推手器，骑行式下肢练习器，手摇式上肢练习器，告示牌	踝关节屈伸训练器立柱底部腐蚀	1
226	\N	\N	拼装地板	民治街道金地上塘道一期5栋	段小姐	18569665058	\N	120平方	健身路径	\N	杂牌	1	7	2012年	26	民治街道金地上塘道一期5栋	[]	LH	3	告示牌，蹬力器，三位扭腰器，腹肌板，太极推手器，钟摆器，滚桶，	二位蹬力器座椅脱漆、盖帽缺失；三位扭腰器盖帽缺失，太极揉推器手柄缺失三个	4
227	\N	\N	拼装地板	民治街道金地上塘道二期10栋	段小姐	18569665058	\N	80平方	健身路径	\N	杂牌	1	6	2014年	26	民治街道金地上塘道二期10栋	[]	LH	3	太空漫步机，太极推手器，蹬力器，腹肌板，推举训练器，腰背按摩器，	太空漫步机摆腿缺失，腹肌板腐蚀	2
228	\N	\N	EPDM	民治街道中航天逸花园平台A2栋	段小姐	18569665058	\N	120平方	健身路径	\N	杂牌	1	7	2010年	26	民治街道中航天逸花园平台A2栋	[]	LH	3	压腿器，单杠，骑马器，椭圆机，骑马器，双杆，	平步机底座腐蚀报废	1
229	\N	\N	地砖	民治街道绿景路边公园	段小姐	18569665058	\N	40平方	健身路径	\N	澳瑞特	1	7	2012年	26	民治街道绿景路边公园	[]	LH	3	椭圆机，三位扭腰器，蹬力器，太空漫步机，骑马器，腰背按摩器，告示牌，	二位蹬力器两边限位损坏、太空漫步机扶手断裂、立柱松动	3
230	\N	\N	/	民治街道幸福风景花园	段小姐	18569665058	\N	/	健身路径	\N	/	1	0	2010年	26	民治街道幸福风景花园	[]	LH	3	已拆除	/	0
231	\N	\N	/	民治街道松仔园篮球场	段小姐	18569665058	\N	/	健身路径	\N	/	1	0	2009年	26	民治街道松仔园篮球场	[]	LH	3	已拆除	/	0
232	\N	\N	/	民治街道松仔园	段小姐	18569665058	\N	/	健身路径	\N	/	1	0	2010年	26	民治街道松仔园	[]	LH	3	已拆除	/	0
233	\N	\N	丙烯酸	横岭一区篮球场53栋	张生	13714678598	\N	700平方	篮球架	\N	金陵	1	1	2019年	27	横岭一区篮球场53栋	[]	LH	3	篮球场	两个篮板缺失防撞条，球网破损一个	2
234	\N	\N	拼装地板	民新工作站横岭小区12栋	张生	13714678598	\N	90平方	健身路径	\N	好家庭	1	12	2017年	27	民新工作站横岭小区12栋	[]	LH	3	告示牌，上肢牵引器，肋木架，太空漫步机，棋牌桌，三位扭腰器，按摩揉推器，腹肌板，骑马器，腰背按摩器，太极揉推器，	正常	0
235	\N	\N	/	民治街道横岭四区	张生	13714678598	\N	/	健身路径	\N	/	1	0	2012年	27	民治街道横岭四区	[]	LH	3	已拆除	已拆除	0
236	\N	\N	拼装地板	碧水龙庭7栋4单元架空层1号+2C栋架空层	张生	13714678598	\N	130平方	健身路径	\N	澳瑞特、好家庭	1	17	2023年	27	碧水龙庭7栋4单元架空层1号+2C栋架空层	[]	LH	3	太空漫步机，三位扭腰器，太极推手器，大转轮，腰背按摩器，骑马器，自重式下压训练器，棋牌桌，太极揉推器，告示牌，多功能训练器，腰背按摩器，太空漫步机，上肢牵引器，三位扭腰器，伸背器，蹬力压腿训练器	正常	0
237	\N	\N	拼装地板	民新社区碧水龙庭4栋旁	张生	13714678598	\N	130平方	力量1、常规2、健身路径	\N	好家庭	1	17	2019年	27	民新社区碧水龙庭4栋旁	[]	LH	3	扩胸训练器，坐式蹬力训练器，推举训练器，坐式前推训练器，腿部训练器，告示牌，骑马器，三位扭腰器，腹肌板，太空漫步机，伸背器，按摩揉推器，四级压腿器，腰背按摩器，大转轮，太极推手器，告示牌	扩胸训练器轴承损坏，	1
238	\N	\N	拼装地板	民新社区碧水龙庭10栋架空层	张生	13714678598	\N	30平方	健身路径	\N	好家庭	1	4	2019年	27	民新社区碧水龙庭10栋架空层	[]	LH	3	按摩揉推器，伸背器，三位扭腰器，腰背按摩器	正常	0
239	\N	\N	拼装地板	民治街道碧水龙庭5栋架空层	张生	13714678598	\N	260平方	健身路径	\N	好家庭	1	35	2016年	27	民治街道碧水龙庭5栋架空层	[]	LH	3	腹肌板，上肢牵引器，肋木架，蹬力器，四级压腿器，太极推手器，骑马器，太空漫步机，告示牌，告示牌，腰背按摩器，太极推手器，伸背器，太空漫步机，蹬力器，三位扭腰器，四级压腿器，上肢牵引器，按摩揉推器，肋木架，腹肌板，蹬力器，棋牌桌，告示牌，告示牌，腹肌板，太极推手器，大转轮，腰背按摩器，按摩揉推器，四级压腿器，伸背器，骑马器，三位扭腰器，太空漫步机，	三件健骑机限位损坏，二位蹬力器限位损坏，天空漫步机盖帽缺失	3
240	\N	\N	丙烯酸	民治街道潜龙花园4栋	张生	13714678598	\N	700平方	篮球场	\N	金陵	1	1	2012年	27	民治街道潜龙花园4栋	[]	LH	3	篮球场	两个篮板防撞边条缺失，地面破损6平方	2
262	\N	\N	安全地垫	民治街道沙吓村大厦旁	舒畅	18998916326	\N	40平方	健身路径	\N	澳瑞特	1	12	2015年	29	民治街道沙吓村大厦旁	[]	LH	3	肋木架，太空漫步机，椭圆机，腹肌板。弹振压腿器，扭腰器，伸背器，太极揉推器，腰背按摩器，按摩揉推器，蹬力器，上肢牵引器	正常	0
241	\N	\N	拼装地板	民治街道潜龙花园7栋	张生	13714678598	\N	150平方	健身路径	\N	体之杰、好家庭	1	22	2016年	27	民治街道潜龙花园7栋	[]	LH	3	告示牌，三位扭腰器，大转轮，太极推手器，三位扭腰器，腰背按摩器，伸背器，双杠，蹬力器，太空漫步机，棋牌桌，伸背器，腹肌板，四级压腿器，腰背按摩器，按摩揉推器，告示牌，上肢牵引器，蹬力器，太极推手器，骑马器，肋木架，	大转轮缺失一边转轮，太极揉推器手柄缺失一个、轴承损坏两个；太空漫步机盖帽缺失一个，二位蹬力器限位损坏两边，健骑机限位损坏，象棋桌坐凳缺失三个；	7
242	\N	\N	拼装地板	民治街道榕苑5栋	张生	13714678598	\N	100平方	健身路径	\N	好家庭	1	15	2019年	27	民治街道榕苑5栋	[]	LH	3	告示牌，象棋桌，棋牌桌，伸背器，四级压腿器，按摩揉推器，太极推手器，腰背按摩器，腹肌板，太空漫步机，骑马器，三位扭腰器，蹬力器，肋木架，上肢牵引器，	二位蹬力器两边限位损坏，健骑机限位损坏，太空漫步机盖帽缺失	3
243	\N	\N	拼装地板	民新工作站碧水龙庭6栋1单元后面	张生	13714678598	\N	100平方	健身路径	\N	好家庭	1	12	2018年	27	民新工作站碧水龙庭6栋1单元后面	[]	LH	3	大转轮，棋牌桌，四级压腿器，告示牌，按摩揉推器，腰背按摩器，伸背器，太极推手器，三位扭腰器，腹肌板，棋牌桌，太空漫步机，	健骑机限位损坏	1
244	\N	\N	拼装地板	民新工作站鑫茂花园c区1栋	张生	13714678598	\N	130平方	健身路径	\N	好家庭	1	12	2017年	27	民新工作站鑫茂花园c区1栋	[]	LH	3	告示牌，上肢牵引器，肋木架，单杠，腰背按摩器，按摩揉推器，三位扭腰器，太极推手器，四级压腿器，骑马器，腹肌板，棋牌桌，	正常	0
245	\N	\N	拼装地板	民新工作站鑫茂花园b区1栋	张生	13714678598	\N	110平方	健身路径	\N	澳瑞特	1	11	2024年	27	民新工作站鑫茂花园b区1栋	[]	LH	3	告示牌，伸背器，蹬力器，自重式下压训练器，太空漫步机，三位扭腰器，转手器，腰背按摩器，太极揉推器，多功能按摩器，上肢牵引器，	正常	0
246	\N	\N	拼装地板	民新工作站龙岸花园一期1栋和3栋架空层	张生	13714678598	\N	230平方	健身路径	\N	好家庭	1	25	2017年	27	民新工作站龙岸花园一期1栋和3栋架空层	[]	LH	3	上肢牵引器，蹬力器，太极推手器，腰背按摩器，肋木架，棋牌桌，伸背器，按摩揉推器，四级压腿器，腹肌板，三位扭腰器，骑马器，太空漫步机，告示牌，，腹肌板，骑马器，肋木架，太空漫步机，上肢牵引器，告示牌，四级压腿器，腰背按摩器，太极推手器，按摩揉推器，三位扭腰器，	按摩揉推器转盘损坏，健骑机限位损坏，上肢牵引器轴承卡住	3
247	\N	\N	地砖	民治街道星河丹堤F区4-3一楼和B区1-1楼架空层	潘嘉城	13148766698	\N	50平方	健身路径	\N	杂牌	1	8	2007年	28	民治街道星河丹堤F区4-3一楼和B区1-1楼架空层	[]	LH	3	太空漫步机，三位扭腰器，三位扭腰器，太空漫步机，太空漫步机，三位扭腰器，骑马器，椭圆机	三位扭腰器转盘轴承损坏三个	1
248	\N	\N	安全地垫	滢水山庄一区22栋	潘嘉城	13148766698	\N	120平方	健身路径	\N	好家庭	1	14	2016年	28	滢水山庄一区22栋	[]	LH	3	单杠，双杠，腰背按摩器，三位扭腰器，按摩揉推器，太空漫步机，腹肌板，肋木架，骑马器，太极推手器，肋木架，伸背器，上肢牵引器，告示牌，	三位扭腰器转盘卡住，双杠盖帽缺失，地垫老旧	3
249	\N	\N	硅PU	滢水山庄一区门口外面小公园	潘嘉城	13148766698	\N	500平方	篮球场	\N	不详	1	1	2015年	28	滢水山庄一区门口外面小公园	[]	LH	3	篮球场	篮球缺失一个，围网破损，地面磨损70平方	3
250	\N	\N	/	滢水山庄二区	潘嘉城	13148766698	\N	/	羽毛球场	\N	/	1	0	2014年	28	滢水山庄二区	[]	LH	3	私人球场	私人球场	0
251	\N	\N	拼装地板	滢水山庄一区门口外面小公园	潘嘉城	13148766698	\N	80平方	健身路径	\N	澳瑞特	1	11	2023年	28	滢水山庄一区门口外面小公园	[]	LH	3	告示牌，太极揉推器，腰背按摩器，上肢牵引器，多功能训练器，自重式下压训练器，伸背器，三位扭腰器，太空漫步机，蹬力压腿训练器，骑马器，	正常	0
252	\N	\N	拼装地板	民乐社区公园	潘嘉城	13148766698	\N	60平方	健身路径	\N	澳瑞特	1	11	2023年	28	民乐社区公园	[]	LH	3	腰背按摩器，太空漫步机，三位扭腰器，多功能训练器，上肢牵引器，自重式下压训练器，骑马器，伸背器，太极揉推器，蹬力压腿训练器，告示牌	三位扭腰器转盘轴承损坏两个	1
253	\N	\N	EPDM	民治街道民乐一区199栋	潘嘉城	13148766698	\N	35平方	健身路径	\N	杂牌	1	5	2013年	28	民治街道民乐一区199栋	[]	LH	3	钟摆器，蹬力器，腰背按摩器，太空漫步机，骑马器，	钟摆器整件报废（轴承损坏、另一边摆腿缺失)；二位蹬力器整件报废	2
254	\N	\N	拼装地板	民治街道丰泽湖小区	潘嘉城	13148766698	\N	120平方	健身路径	\N	好家庭	1	13	2016年	28	民治街道丰泽湖小区	[]	LH	3	肋木架，太极揉推器，上肢牵引器，四级压腿器，蹬力器，腰背按摩器，按摩揉推器，三位扭腰器，腹肌板，骑马器，太空漫步机，伸背器，告示牌	二位蹬力器两边限位损坏，健骑机限位损坏，太空漫步机立柱松动、盖帽缺失	4
255	\N	\N	地砖	民治街道溪山美地一期13栋	潘嘉城	13148766698	\N	80平方	健身路径	\N	童瑶、澳瑞特	1	12	2014年	28	民治街道溪山美地一期13栋	[]	LH	3	腰背按摩器，按摩揉推器，单杠，告示牌，太极揉推器，太极推手器，蹬力器，太空漫步机，三位扭腰器，跷跷板，腹肌板，健身车	正常	0
256	\N	\N	EPDM	民治街道溪山美地三期3栋	潘嘉城	13148766698	\N	35平方	健身路径	\N	童瑶	1	5	2014年	28	民治街道溪山美地三期3栋	[]	LH	3	蹬力器，太空推手器，骑马器，单杠，太极揉推器，	正常	0
257	\N	\N	地砖、EPDM	民治街道溪山美地三期6栋	潘嘉城	13148766698	\N	40平方	健身路径	\N	杂牌	1	6	2014年	28	民治街道溪山美地三期6栋	[]	LH	3	太极推手器，椭圆机，肋木架，双杠，太极推手器，椭圆机，	正常	0
258	\N	\N	/	民治街道溪山美地4	潘嘉城	13148766698	\N	/	健身路径	\N	/	1	0	2014年	28	民治街道溪山美地4	[]	LH	3	已更新	已更新	0
259	\N	\N	安全地垫	民治街道骏景华庭c栋	舒畅	18998916326	\N	70平方	健身路径	\N	好家庭	1	11	2016年	29	民治街道骏景华庭c栋	[]	LH	3	太空漫步机，骑马器，肋木架，腰背按摩器，肋木架，三位扭腰器，太极推手器，按摩揉推器，伸背器，告示牌，腹肌板，	腰背按摩器滚轮缺失	1
260	\N	\N	/	民治街道沙元埔66栋	舒畅	18998916326	\N	/	健身路径	\N	/	1	0	2017年	29	民治街道沙元埔66栋	[]	LH	3	已拆除	/	0
263	\N	\N	草地、拼装地板	民治街道梅花山庄c47栋	舒畅	18998916326	\N	150平方	健身路径	\N	好家庭/澳瑞特	1	17	2015年	29	民治街道梅花山庄c47栋	[]	LH	3	扩胸训练器，推举训练器，棋牌桌，腰背按摩器，伸背器，蹬力器，单杠，三位扭腰器，椭圆机，告示牌，按摩揉推器，肋木架，双杠，太极推手器，太极揉推器，多功能训练器，太空漫步机，	推举训练器限位损坏，护胸训练器限位损坏，	2
264	\N	\N	丙烯酸	民治街道馨园一区游泳池旁	舒畅	18998916326	\N	500平方	篮球场	\N	金陵	1	1	2010年	29	民治街道馨园一区游泳池旁	[]	LH	3	篮球场	球网破损一个，地面磨损严重	2
265	\N	\N	拼装地板	民治街道馨园一期游泳池旁	舒畅	18998916326	\N	80平方	健身路径	\N	澳瑞特	1	8	2017年	29	民治街道馨园一期游泳池旁	[]	LH	3	告示牌，天梯，双杠，肋木架，单杠，太空漫步机，小双杠，弹振压腿器，	正常	0
266	\N	\N	地砖	民治工作站馨园二期别墅30栋	舒畅	18998916326	\N	130平方	健身路径	\N	好家庭	1	14	2018年	29	民治工作站馨园二期别墅30栋	[]	LH	3	太极推手器，太空漫步机，按摩揉推器，大转轮，棋牌桌，三位扭腰器，腰背按摩器，腰背按摩器，椭圆机，腹肌板，骑马器，伸背器，四级压腿器，告示牌，	正常	0
267	\N	\N	安全地垫	民治街道馨园二期7栋旁	舒畅	18998916326	\N	240平方	健身路径	\N	桂宇星、好家庭	1	26	2012年	29	民治街道馨园二期7栋旁	[]	LH	3	告示牌，象棋桌，按摩揉推器，腰背按摩器，太极推手器，骑马器，上肢牵引器，肋木架，太空漫步机，三位扭腰器，腹肌板，双杠，肋木架，跷跷板，大转轮，蹬力器，告示牌，腹肌板，小双杠，单杠，伸背器，腰背按摩器，腹肌板，太空漫步机，太极推手器，压腿器	腰背按摩器盖帽缺失，地垫缺失60平方	2
268	\N	\N	丙烯酸	民治街道社区公园	舒畅	18998916326	\N	750平方	篮球场	\N	金陵	1	1	2016年	29	民治街道社区公园	[]	LH	3	篮球场	篮板防撞条损坏	1
269	\N	\N	/	吴治街道德爱电子有限公司球埸	舒畅	18998916326	\N	/	篮球场	\N	/	1	0	2015年	29	吴治街道德爱电子有限公司球埸	[]	LH	3	倒闭	/	0
270	\N	\N	拼装地板	龙华区民治街道白石龙社区汇龙苑6栋	曾俏玲	13715227547	\N	80平方	儿童娱乐设施	\N	好家庭	1	1	2019件	30	龙华区民治街道白石龙社区汇龙苑6栋	[]	LH	3	儿童滑滑梯	正常	0
271	\N	\N	拼装地板	民新工作站汇龙苑中心花园3栋	曾俏玲	13715227547	\N	40平方	健身路径	\N	好家庭	1	6	2017年	30	民新工作站汇龙苑中心花园3栋	[]	LH	3	上肢牵引器，肋木架，腰背按摩器，四级压腿器，三位扭腰器，太空漫步机	正常	0
272	\N	\N	安全地垫	民治街道阳光新苑3C栋	曾俏玲	13715227547	\N	100平方	健身路径	\N	澳瑞特、体之杰	1	19	2015年	30	民治街道阳光新苑3C栋	[]	LH	3	告示牌，双杠，双人坐推，蹬力器，双人坐推，单杠，棋牌桌，伸背器，太极揉推器，太空漫步机，腰背按摩器，椭圆机，三位扭腰器，斜躺式健身车，仰卧起坐板，按摩揉推器，象棋桌，骑马器，告示牌，	太空漫步机轴承损坏	1
273	\N	\N	丙烯酸	中航阳光新苑西北门外	曾俏玲	13715227547	\N	1200平方	篮球场	\N	好家庭	1	2	2020年	30	中航阳光新苑西北门外	[]	LH	3	篮球场	篮网缺失两个，篮板防撞边条缺失一条，地面开裂	3
274	\N	\N	安全地垫	民治街道逸秀新村73栋	曾俏玲	13715227547	\N	70平方	健身路径	\N	杂牌	1	8	2014年	30	民治街道逸秀新村73栋	[]	LH	3	告示牌，三位扭腰器，上肢牵引器，太极揉推器，太空漫步机，腰背按摩器，腹肌板，棋牌桌，	正常	0
275	\N	\N	拼装地板	民治街道汇龙苑6栋	曾俏玲	13715227547	\N	120平方	健身路径	\N	好家庭	1	20	2017年9月	30	民治街道汇龙苑6栋	[]	LH	3	太极推手器，棋牌桌，腹肌板，按摩揉推器，骑马器，告示牌，按摩揉推器，四级压腿器，上肢牵引器，告示牌，伸背器，蹬力器，腹肌板，骑马器，太空漫步机，肋木架，三位扭腰器，太极推手器，腰背按摩器，棋牌桌，	二位蹬力器限位损坏，太空漫步机轴承损坏	2
276	\N	\N	EPDM	民治街道所白石龙一区195栋	曾俏玲	13715227547	\N	200平方	健身路径	\N	杂牌	1	13	2013年	30	民治街道所白石龙一区195栋	[]	LH	3	压腿器，双人坐推，太空漫步机，太空漫步机，太极推手器，太极推手器，腰背按摩器，蹬力器，三位扭腰器，三位扭腰器，伸背器，告示牌，椭圆机，	二位蹬力器报废，太空漫步机轴承损坏，健骑机两边轴承固定螺丝缺失，三位扭腰器转盘缺失三个，三位扭腰器转盘损坏，平步机底部固定螺丝缺失	6
277	\N	\N	拼装地板	民治街道白石龙二区44栋	曾俏玲	13715227547	\N	260平方	健身路径	\N	杂牌	1	23	2012年	30	民治街道白石龙二区44栋	[]	LH	3	伸背器，三位扭腰器，太空漫步机，骑马器，蹬力器，双杠，太极推手器，蹬力器，双杠，三位扭腰器，太空漫步机，太空漫步机，伸背器，三位扭腰器，伸背器，太空漫步机，太空漫步机，双杠，三位扭腰器，蹬力器，骑马器，太极推手器，太极揉推器，	两件三位扭腰器转盘缺失一个；太空漫步机轴承损坏、盖帽缺失两个、扶手脱焊；二位蹬力器坐凳损坏、盖帽缺失；两件双杠固定螺丝缺失，太空漫步机扶手脱焊，三位扭腰器转盘损坏两个，二位蹬力器盖帽缺失；二位蹬力器坐凳损坏，三位扭腰器转盘损坏一个，	9
278	\N	\N	/	白石龙望辉路公园	曾俏玲	13715227547	\N	/	篮球场	\N	/	1	0	2017年	30	白石龙望辉路公园	[]	LH	3	已拆除	已拆除	0
279	\N	\N	丙烯酸	民治街道东一村东北门	李女士	13691913904	\N	500平方	篮球场	\N	不详	1	1	2010年	31	民治街道东一村东北门	[]	LH	3	篮球场	地面破损10平方	1
280	\N	\N	拼装地板	民治街道西头新村党群服务中心旁	李女士	13691913904	\N	110平方	健身路径	\N	澳瑞特	1	11	2024年	31	民治街道西头新村党群服务中心旁	[]	LH	3	告示牌，蹬力器，太极揉推器，腰背按摩器，多功能按摩器，上肢牵引器，太空漫步机，三位扭腰器，转手器，伸背器，自重式下压训练器，	正常	0
281	\N	\N	/	民治街道西头村	李女士	13691913904	\N	/	健身路径	\N	/	1	0	2013年	31	民治街道西头村	[]	LH	3	已拆除	/	0
282	\N	\N	EPDM	民治街道西头公园（上芬社区公园）	李女士	13691913904	\N	100平方	健身路径	\N	桂宇星	1	9	2013年	31	民治街道西头公园（上芬社区公园）	[]	LH	3	告示牌，太极推手器，骑马器，腰背按摩器，腹肌板，跷跷板，蹬力器，伸背器，	腰背按摩器滚轮损坏，跷跷板限位损坏，腹肌板腐蚀	3
283	\N	\N	拼装地板、安全地垫	民治街道银泉花园8栋	李女士	13691913904	\N	140平方	健身路径	\N	好家庭	1	17	2019年	31	民治街道银泉花园8栋	[]	LH	3	告示牌，腰背按摩器，大转轮，骑马器，腹肌板，太极推手器，伸背器，四级压腿器，双杠，太极推手器，按摩揉推器，肋木架，大转轮，四级压腿器，太空漫步机，三位扭腰器，腰背按摩器，	健骑机限位损坏，	1
284	\N	\N	安全地垫	民治街道银泉花园羽毛球场旁	李女士	13691913904	\N	35平方	健身路径	\N	杂牌	1	3	2018年	31	民治街道银泉花园羽毛球场旁	[]	LH	3	腹肌板，三位扭腰器，太空漫步机，	安全地垫磨损老旧	1
285	\N	\N	拼装地板	上芬工作站银泉花园9栋旁	李女士	13691913904	\N	40平方	健身路径	\N	好家庭	1	7	2018年	31	上芬工作站银泉花园9栋旁	[]	LH	3	骑马器，伸背器，太空漫步机，按摩揉推器，三位扭腰器，腹肌板，告示牌	正常	0
286	\N	\N	/	榕树苑	李女士	13691913904	\N	/	健身路径	\N	/	1	0	2019年	31	榕树苑	[]	LH	3	已拆除	已拆除	0
287	\N	\N	/	榕树苑	李女士	13691913904	\N	/	常规1	\N	/	1	0	\N	31	榕树苑	[]	LH	3	已拆除	已拆除	0
288	\N	\N	水磨石地面	上芬工作站玉华花园篮球场c栋	李女士	13691913904	\N	500平方	篮球架	\N	金陵	1	1	2018年	31	上芬工作站玉华花园篮球场c栋	[]	LH	3	篮球架	篮网缺失一个，篮板防撞边条缺失	2
289	\N	\N	拼装地板	玉华花园c栋	李女士	13691913904	\N	120平方	健身路径	\N	好家庭	1	12	2016年	31	玉华花园c栋	[]	LH	3	太空漫步机，告示牌，四级压腿器，按摩揉推器，棋牌桌，腰背按摩器，太极推手器，三位扭腰器，伸背器，肋木架，上肢牵引器，蹬力器，	太空漫步机盖帽缺失，二位蹬力器两边限位损坏	2
290	\N	\N	/	民治街道龙塘公园	段佳佳	29787502	\N	/	健身路径	\N	/	1	0	2013年	32	民治街道龙塘公园	[]	LH	3	已拆除	已拆除	0
291	\N	\N	EPDM	民治街道长城里程家园	段佳佳	29787502	\N	180平方	健身路径	\N	好家庭、澳瑞特	1	21	2010年	32	民治街道长城里程家园	[]	LH	3	裸关节屈伸练习器，转手器，肩梯/上肢协调功能练习器，背部伸展/腕关节/提力练习器，肩关节回旋/前臂、腕关节练习器，双杠，环形上肢协调功能练习器，骑行式下肢练习器，肋木架，肋木架，单杆，腰背按摩器，太空漫步机，三位扭腰器，按摩揉推器，太极推手器，骑马器，伸背器，腹肌板，告示牌，平行杠步态练习器	踝关节屈伸训练器报废，手摇式上肢练习器报废，前臂腕关节练习器手柄缺失；太空漫步机摆腿缺失	4
292	\N	\N	拼装地板	民治街道中海锦城	段佳佳	29787502	\N	40平方	健身路径	\N	杂牌	1	5	2013年	32	民治街道中海锦城	[]	LH	3	骑马器，伸背器，腰背按摩器，太空漫步机，大转轮	正常	0
293	\N	\N	安全地垫	民治街道远景家园	段佳佳	29787502	\N	70平方	健身路径	\N	杂牌	1	3	2014年	32	民治街道远景家园	[]	LH	3	三位扭腰器，双杠，太空漫步机	正常	0
294	\N	\N	丙烯酸	民治街道简上村二区52栋篮球场	段佳佳	29787502	\N	500平方	篮球场	\N	不详	1	1	2012年	32	民治街道简上村二区52栋篮球场	[]	LH	3	篮球场	篮球架报废一套，篮筐缺失一个，地面开裂磨损	3
295	\N	\N	地砖	民治街道星河传奇一期1栋	段佳佳	29787502	\N	40平方	健身路径	\N	杂牌	1	5	2014年	32	民治街道星河传奇一期1栋	[]	LH	3	太极揉推器，太空漫步机，腰背按摩器，太空漫步机，太极揉推器	正常	0
296	\N	\N	EPDM	民治街道简上社区公园	段佳佳	29787502	\N	200平方	健身路径	\N	杂牌	1	9	2014年	32	民治街道简上社区公园	[]	LH	3	跷跷板，蹬力器，压腿器，双杠，太极揉推器，腰背按摩器，太空漫步机，三位扭腰器	正常	0
297	\N	\N	EPDM	民治街道皇后道4栋	段佳佳	29787502	\N	120平方	健身路径	\N	杂牌	1	13	2012年	32	民治街道皇后道4栋	[]	LH	3	椭圆机，跑步机，腰背按摩器，跷跷板，腹肌板，单杠，双杠，太极推手器，跷跷板，跷跷板，前推训练器，三位扭腰器，三位扭腰器，	跷跷板塑料座垫缺失，推举训练器盖帽缺失、立柱松动，三位扭腰器转盘缺失一个，三位扭腰器转盘损坏两个，腰背按摩器立柱松动，两件跷跷板立柱松动	6
298	\N	\N	地砖	民治街道御龙华庭3栋	段佳佳	29787502	\N	60平方	健身路径	\N	杂牌	1	7	2012年	32	民治街道御龙华庭3栋	[]	LH	3	骑马器，太空漫步机，三位扭腰器，双杠，钟摆器，椭圆机，跷跷板	跷跷板限位损坏	1
299	\N	\N	EPDM	民治街道龙光玖悦台1栋	段佳佳	29787502	\N	70平方	健身路径	\N	杂牌	1	8	2017年	32	民治街道龙光玖悦台1栋	[]	LH	3	蹬力器，钟摆器，椭圆机，三位扭腰器，骑马器，椭圆机，钟摆器，划船器	椭圆机报废	1
300	\N	\N	拼装地板	民治街道星河盛世A1栋	徐观平	13534171844	\N	400平方	健身路径	\N	杂牌	1	7	2012年	33	民治街道星河盛世A1栋	[]	LH	3	太极揉推器，蹬力器，天梯，双杠，伸背器，扭腰步道，腰背按摩器，	腰背按摩器滚轮缺失，太极揉推器转盘缺失两个	2
301	\N	\N	EPDM	民治街道星河盛世A2栋	徐观平	13534171844	\N	40平方	健身路径	\N	杂牌	1	3	2012年	33	民治街道星河盛世A2栋	[]	LH	3	肋木架，双杠，单杠	正常	0
302	\N	\N	拼装地板	民治街道星河盛世A3栋	徐观平	13534171844	\N	6平方	健身路径	\N	杂牌	1	1	2012年	33	民治街道星河盛世A3栋	[]	LH	3	组合训练器	正常	0
303	\N	\N	拼装地板	民治街道鑫海公寓A栋	徐观平	13534171844	\N	110平方	健身路径	\N	好家庭	1	14	2016年	33	民治街道鑫海公寓A栋	[]	LH	3	上肢牵引器，肋木架，蹬力器，按摩揉推器，腰背按摩器，太极推手器，四级压腿器，三位扭腰器，骑马器，告示牌，腹肌板，伸背器，棋牌桌，太空漫步机，太空漫步机	二位蹬力器限位损坏，太空漫步机轴承损坏	2
304	\N	\N	安全地垫、拼装地板	民泰工作站书香门第上河坊1-B座	徐观平	13534171844	\N	130平方	健身路径	\N	好家庭/澳瑞特	1	24	2018年	33	民泰工作站书香门第上河坊1-B座	[]	LH	3	伸背器，告示牌，椭圆机，三位扭腰器，太极揉推器，太空漫步机，伸背器，仰卧起坐板，象棋桌，棋牌桌，大转轮，弹振训练器，多功能训练器，腰背按摩器，按摩揉推器，蹬力器，太空漫步机，四级压腿器，骑马器，太极推手器，腹肌板，按摩揉推器，腰背按摩器，告示牌	太极揉推器转盘轴承损坏两个	1
329	\N	\N	拼装地板	丹坑园区26栋旁	张伟龙	13714808911	\N	130	健身路径	\N	杰威2023	1	11	2023年	38	丹坑园区26栋旁	[]	LH	4	腹肌板、伸背器、太极揉推器、上肢牵引器、二位蹬力器、肋木架、健骑机、太空漫步机、腰背按摩器、告示牌、三位扭腰器	正常	0
306	\N	\N	安全地垫	民治街道书香门第名苑8栋	徐观平	13534171844	\N	80平方	健身路径	\N	澳瑞特	1	15	2014年	33	民治街道书香门第名苑8栋	[]	LH	3	告示牌，斜躺式健身车，仰卧起坐板，象棋桌，椭圆机，三位扭腰器，多功能训练器，太极揉推器，伸背器，弹振训练器，单杠，按摩揉推器，腰背按摩器，太空漫步机，蹬力器	正常	0
307	\N	\N	拼装地板	民治街道翠岭华庭3栋	徐观平	13534171844	\N	50平方	健身路径	\N	好家庭	1	10	2016年	33	民治街道翠岭华庭3栋	[]	LH	3	棋牌桌，上肢牵引器，肋木架，伸背器，腹肌板，骑马器，三位扭腰器，太空漫步机，告示牌，蹬力器，	二位蹬力器两边限位损坏，三位扭腰器转盘缺失一个	2
308	\N	\N	EPDM	民治街道万科金域华府1期8栋	徐观平	13534171844	\N	260平方	健身路径	\N	杂牌、澳瑞特	1	11	2010年	33	民治街道万科金域华府1期8栋	[]	LH	3	蹬力器，太空漫步机，椭圆机，腹肌板，腰背按摩器，蹬力器，四级压腿器，太空漫步机，单杠，三位扭腰器，椭圆机	四位蹬力器坐凳缺失	1
309	\N	\N	EPDM	民治街道万科金域华府2期3栋	徐观平	13534171844	\N	60平方	健身路径	\N	天健体育、澳瑞特	1	6	2010年	33	民治街道万科金域华府2期3栋	[]	LH	3	上肢牵引器，腰背按摩器，单杠，双杠，三位扭腰器，椭圆机	上肢牵引器缺失两根牵绳	1
310	\N	\N	安全地垫	民治街道世纪春城三期6号地1栋2楼	苏韦宇	13510260098	\N	30平方	健身路径	\N	杂牌	1	2	2023年	34	民治街道世纪春城三期6号地1栋2楼	[]	LH	3	已拆除	已拆除	0
311	\N	\N	安全地垫	民治街道世纪春城三期八号地1栋	苏韦宇	13510260098	\N	12平方	健身路径	\N	澳瑞特	1	2	2011年	34	民治街道世纪春城三期八号地1栋	[]	LH	3	斜躺式健身车，椭圆机	正常	0
312	\N	\N	拼装地板	民治街道世纪春城三期3栋	苏韦宇	13510260098	\N	170平方	健身路径	\N	杂牌	1	15	2023年	34	民治街道世纪春城三期3栋	[]	LH	3	告示牌，太极推手器，按摩揉推器，仰卧起坐训练器，四级压腿器，上肢牵引器，单杠，多功能训练器，太空漫步机，天梯，椭圆机，骑马器，三位扭腰器，蹬力器，腰背按摩器，	二位蹬力器两边限位损坏，太空漫步机摆腿脱焊	2
313	\N	\N	EPDM	民治街道世纪春城四期6栋	苏韦宇	13510260098	\N	60平方	健身路径	\N	好家庭/澳瑞特	1	5	2012年	34	民治街道世纪春城四期6栋	[]	LH	3	棋牌桌，象棋桌，单杠，太空漫步机，腰背按摩器，	围棋桌座椅立柱底部腐蚀	1
314	\N	\N	地砖	民治街道东边老村22栋旁	苏韦宇	13510260098	\N	80平方	健身路径	\N	杂牌	1	10	2016年	34	民治街道东边老村22栋旁	[]	LH	3	上肢牵引器，上肢牵引器，三位扭腰器，三位扭腰器，太空漫步机，太空漫步机，钟摆器，腰背按摩器，钟摆器，腰背按摩器	太空漫步机盖帽破损三个	1
315	\N	\N	/	民治水尾村的楼顶球场	苏韦宇	13510260098	\N	/	篮球架	\N	/	1	0	2019年	34	民治水尾村的楼顶球场	[]	LH	3	已拆除	已拆除	0
316	\N	\N	/	民治水尾村的楼顶球场	苏韦宇	13510260098	\N	/	篮球架	\N	/	1	0	\N	34	民治水尾村的楼顶球场	[]	LH	3	已拆除	已拆除	0
317	\N	\N	安全地垫	民治街道春华四季园9栋后面	沈军	13530814829	\N	135平方	健身路径	\N	好家庭	1	15	2013年	35	民治街道春华四季园9栋后面	[]	LH	3	告示牌，双杠，蹬力器，太极推手器，单杠，腰背按摩器，伸背器，太空漫步机，象棋桌，肋木架，三位扭腰器，按摩揉推器，腹肌板，上肢牵引器，肋木架，	太空漫步机轴承损坏，象棋桌面板脱落，太极揉推器手柄缺失，	3
318	\N	\N	EPDM	民治街道春华四季园39栋旁	沈军	13530814829	\N	100平方	健身路径	\N	好家庭、澳瑞特	1	14	2015年	35	民治街道春华四季园39栋旁	[]	LH	3	告示牌，单杠，太空漫步机，三位扭腰器，椭圆机，斜躺式健身车，棋牌桌，告示牌，按摩揉推器，腰背按摩器，太极揉推器，伸背器，仰卧起坐板，象棋桌，	正常	0
319	\N	\N	丙烯酸	民治街道春华四季园39栋	沈军	13530814829	\N	500平方	篮球场	\N	金陵	1	1	2012年	35	民治街道春华四季园39栋	[]	LH	3	篮球场	篮板防撞条缺失一条，地面开裂	2
320	\N	\N	地砖	民治街道浩月花园3栋	沈军	13530814829	\N	120平方	健身路径	\N	奥特康、万德、体之杰、杂牌	1	15	2008年	35	民治街道浩月花园3栋	[]	LH	3	太空漫步机，告示牌，腹肌板，三位扭腰器，压腿器，太极推手器，腹肌板，太空漫步机，太极推手器，腰背按摩器，四级压腿器，太空漫步机，太空漫步机，三位扭腰器，太极漫步机，	太空漫步机盖帽缺失	1
321	\N	\N	丙烯酸	民治街道民新工业区	沈军	13530814829	\N	800平方	篮球场	\N	金陵	1	3	2012年	35	民治街道民新工业区	[]	LH	3	篮球场	地面破损	1
322	\N	\N	/	红木山水厂内	林先生	19129355107	\N	/	网球场	\N	/	1	0	2021年	36	红木山水厂内	[]	LH	3	不让进	不让进	0
323	\N	\N	/	民治街道大岭社区红木山水厂内	林先生	19129355107	\N	/	网球场	\N	/	1	0	2022-03-08	36	民治街道大岭社区红木山水厂内	[]	LH	3	不让进	不让进	0
324	\N	\N	拼装地板	1866花园南区架空层C栋	林先生	19129355107	\N	110平方	健身路径	\N	杂牌、澳瑞特	1	15	2023年	36	1866花园南区架空层C栋	[]	LH	3	双杠，蹬力器，三位扭腰器，太空漫步机，伸背器，太空漫步机，三位扭腰器，太极揉推器，按摩揉推器，骑马器，多功能训练器，蹬力压腿训练器，上肢牵引器，自重式下压训练器，告示牌	正常	0
325	\N	\N	地砖、安全地垫	民治街道汇龙湾1、2栋	林先生	19129355107	\N	60平方	健身路径	\N	杂牌	1	7	2013年	36	民治街道汇龙湾1、2栋	[]	LH	3	双杠，单杠，太空漫步机，骑马器，太极推手器，太空漫步机，统计揉推器	双杠报废，	1
326	\N	\N	/	民治街道民康中队	-	15807643792	\N	/	健身路径	\N	/	1	0	2014年	37	民治街道民康中队	[]	LH	3	已拆除	不让进	0
327	\N	\N	拼装地板	福安雅园A区9栋旁	张伟龙	13714808911	\N	300	健身路径	\N	杰威、好家庭	1	25	2022年	38	福安雅园A区9栋旁	[]	LH	4	健骑机x2、伸背器x2、太极揉推器x2、三位扭腰器x2、告示牌x2、腰背按摩器x2、太空漫步机x2、压腿器、单杠、双杠x2、二位蹬力器、伸展器、上肢牵引器x2、手部腿部按摩器、腹肌板x2	两件太空漫步机8个轴承损坏，二位蹬力器座位座板缺失	2
328	\N	\N	丙烯酸	福安雅园C区7栋旁	张伟龙	13714808911	\N	1000	篮球架(固定式)	\N	杂牌	1	2	2019-01-01	38	福安雅园C区7栋旁	[]	LH	4	两片场地	三个篮网损坏，地面破损	2
330	\N	\N	拼装地板	丹坑新村北20栋旁	张伟龙	13714808911	\N	80	健身路径	\N	杰威2023	1	9	2023-01-01	38	丹坑新村北20栋旁	[]	LH	4	腹肌板、伸背器、上肢牵引器、二位蹬力器、肋木架、太空漫步机、腰背按摩器、告示牌、三位扭腰器	正常	0
331	\N	\N	硅pu	丹坑村股份有限公司办公楼前	张伟龙	13714808911	\N	550	篮球架（移动式）	\N	金陵2022	1	2	2022-01-01	38	丹坑村股份有限公司办公楼前	[]	LH	4	2副篮球架	正常	0
332	\N	\N	拼装地板	丰盛懿园A座一楼旁马路边	张伟龙	13714808911	\N	60	健身路径	\N	好家庭2022	1	8	2022-11-29	38	丰盛懿园A座一楼旁马路边	[]	LH	4	腰背按摩器、四级压腿器、太极揉推器、上下肢训练器、太空漫步机、背肌训练器、钟摆器、告示牌	腰背按摩器限位损坏、上肢训练器限位损坏、太极揉推器手柄损坏	3
333	\N	\N	沥青100平方+拼装地板50平方+地砖40平方	丰盛懿园A座三楼架空层、平台	张伟龙	13714808911	\N	沥青100平方+拼装地板50平方+地砖40平方	健身路径	\N	杰威、澳瑞特2022	1	28	2022-12-09	38	丰盛懿园A座三楼架空层、平台	[]	LH	4	腰背按摩器、伸背器、太极揉推器、平步机、太极揉推器、健骑机、告示牌x2、象棋桌x6、坐式划船训练器、腰部侧屈训练器、坐式踢腿训练器、背肌训练器、直立健身车x3、斜躺健身车x3、太极揉推器、腹肌板、围棋桌x2、	正常	0
334	\N	\N	沥青50平方	福水路3号丰盛懿园A座三楼平台	张伟龙	13714808911	\N	50	儿童游乐设施	\N	不详	1	1	2022年	38	福水路3号丰盛懿园A座三楼平台	[]	LH	4	儿童滑梯	滑筒开裂破损	1
335	\N	\N	EPD60平方	观荟名庭3栋B单元1楼平台	张伟龙	13714808911	\N	60	儿童游乐设施	\N	杂牌2022	1	1	2022-01-01	38	观荟名庭3栋B单元1楼平台	[]	LH	4	儿童滑梯	玻璃罩缺失、两根立柱装饰件缺失	2
336	\N	\N	水泥地	宝观科技园	杨庆发	13824323100	\N	130平方	健身路径	\N	康乐达2013年	1	19	2013年	39	宝观科技园	[]	LH	4	上肢牵引器，太空漫步机，蹬力器x2，腰背按摩器x4，钟摆器，三位扭腰器，天梯双杠，直立健身车x2，腹肌板x2，告示牌，单杠x2	两件腹肌板脱漆	2
337	\N	\N	丙烯酸	宝观科技园	杨庆发	13824323100	\N	500平方	篮球架（移动式）	\N	好家庭2013年	1	1	不详	39	宝观科技园	[]	LH	4	篮球架	正常	0
338	\N	\N	\N	章阁科技园	杨庆发	13824323100	\N	\N	篮球架	\N	金陵2019年	1	4	不详	39	章阁科技园	[]	LH	4	篮球架	闲置状态	0
339	\N	\N	\N	福城街道樟阁城市公园森林消防站旁（1）	杨庆发	13824323100	\N	70平方	健身路径	\N	杂牌2017年	1	3	2017年	39	福城街道樟阁城市公园森林消防站旁（1）	[]	LH	4	太空漫步机x2，双杠	地面破损	1
340	\N	\N	拼装地板	福城街道章阁背礼园公园	杨庆发	13824323100	\N	100平方	健身路径	\N	好家庭2015年	1	9	2017年	39	福城街道章阁背礼园公园	[]	LH	4	太空漫步机，太极揉推器x2，腹肌板，腿部按摩器，腰背按摩器，肋木架，二位蹬力器，三位扭腰器，	腹肌板腐蚀报废，太空漫步机脱漆，腿部按摩器脱漆，腰背按摩器脱漆，肋木架脱漆，二位蹬力器脱漆立柱螺丝松动，三位扭腰器转盘缺失两个，	7
341	\N	\N	拼装地板	胡润名苑小区b栋	杨庆发	13824323100	\N	100平方	儿童游乐设施	\N	好家庭（飞友）	1	1	不详	39	胡润名苑小区b栋	[]	LH	4	儿童滑梯	正常	0
342	\N	\N	安全地垫	章阁工业园	杨庆发	13824323100	\N	70平方	健身路径	\N	澳瑞特2019	1	9	2019年	39	章阁工业园	[]	LH	4	告示牌，象棋桌，多功能训练器，扭腰步道，仰卧起坐练习器，健骑机，伸背器，二位蹬力器，太空漫步机，	地垫老旧，象棋桌立柱松动、四个凳子缺失；太空漫步机两个轴承损坏	4
343	\N	\N	硅pu	章阁老村170栋	杨庆发	13824323100	\N	500平方	篮球架	\N	金陵2019年	1	1	不详	39	章阁老村170栋	[]	LH	4	篮球架篮球场	一篮板边条松动、篮筐损坏，另一篮板边条缺失	3
344	\N	\N	地砖	大水坑村大一组碉楼旁	黄先生	15013696660	\N	200平方	健身路径	\N	杂牌2021	1	13	2021年	40	大水坑村大一组碉楼旁	[]	LH	4	太空漫步机x4、太极揉推器x2、三位扭腰器x2、告示牌、椭圆机x3、钟摆器、	钟摆器摆腿断缺、太空漫步机扶手断裂、三件椭圆机轴承损坏	3
345	\N	\N	丙烯酸	大水坑二村94栋	黄先生	15013696660	\N	600平方	篮球场	\N	金陵2022	1	1	2011年	40	大水坑二村94栋	[]	LH	4	篮球架篮球场	篮网破损	1
346	\N	\N	EPDM	大水坑党群活动中心后面	黄先生	15013696660	\N	200平方平方	健身路径	\N	杰威、好家庭、澳瑞特、杂牌，	1	22	2019年	40	大水坑党群活动中心后面	[]	LH	4	告示牌x2、二位蹬力器x2、上肢牵引器、肋木架、三位扭腰器x3、健骑机、太极揉推器x2、腹肌板、四级压腿器、象棋桌x2、伸背器x2、压腿器、腰背按摩器、单杠、鞍马训练器、	告示牌边条损坏、肋木架脱漆、三位扭腰器脱漆、腹肌板腐蚀报废、伸背器脱漆、颗粒地垫老旧破损、太极揉推器三个转盘轴承损坏	7
347	\N	\N	丙烯酸	大水坑党群活动中心街心公园	黄先生	15013696660	\N	700平方	篮球场	\N	好家庭	1	1	2023年	40	大水坑党群活动中心街心公园	[]	LH	4	篮球架篮球场	地面破损、篮网损坏两个	2
348	\N	\N	拼装地板	冠志工业园d2栋	黄先生	15013696660	\N	70平方	健身路径	\N	好家庭2014	1	13	2014年	40	冠志工业园d2栋	[]	LH	4	肋木架、二位蹬力器、腰背按摩器、健骑机、四级压腿器、太极揉推器、太空漫步机、三位扭腰器、手部腿部按摩器、腹肌板、伸背器、上肢牵引器，象棋桌	地板缺失破损8平方	1
349	\N	\N	拼装地板	金富苑b栋	黄先生	15013696660	\N	90平方	健身路径	\N	杰威	1	11	2015年	40	金富苑b栋	[]	LH	4	天梯、直立健身车、上肢牵引器、三位扭腰器、伸背器、二位蹬力器、腰背按摩器、太极揉推器、肋木架、告示牌、仰卧起坐练习器	正常	0
350	\N	\N	拼装地板	福苑暖心社区	黄先生	15013696660	\N	70平方	健身路径	\N	好家庭	1	8	2022-11-29	40	福苑暖心社区	[]	LH	4	腰背按摩器、钟摆器、太空漫步机、上下肢训练器、太极揉推器、四级压腿器、背肌训练器、告示牌	正常	0
351	\N	\N	地砖	九龙山体育公园	黄先生	15013696660	\N	200平方	室外智能健身器材	\N	好家庭	1	24	2021年	40	九龙山体育公园	[]	LH	4	\N	钟摆器摆腿断缺、太空漫步机扶手断裂、三件椭圆机轴承损坏	3
373	\N	\N	安全地垫	观湖街道招商澜园南区10栋	沈娇露	18123765610	\N	130平方	健身路径	\N	桂宇星、好家庭	1	10	2016年	45	观湖街道招商澜园南区10栋	[]	LH	5	双杠x2，二位蹬力器x2，告示牌，太极揉推器，太空漫步机，腹肌板，天梯，压腿器	二位蹬力器限位损坏，二位蹬力器立柱松动限位损坏，太极揉推器手柄缺失，地垫老旧	4
352	\N	\N	地砖40平方+拼装地板140平方	招商锦锈观园6栋旁	陈挺	13751091955	\N	地砖40平方+拼装地板140平方	健身路径	\N	杰威、澳瑞特、杂牌2019	1	21	2019年	41	招商锦锈观园6栋旁	[]	LH	4	太极揉推器x2、压腿器、腰背按摩器x2、大转轮、单杠、双杠、二位蹬力器x2，手脚转动器、三位扭腰器x2、告示牌、伸背器x2、象棋桌x2、伸展器、太空漫步机x2、	象棋桌脱漆、太空漫步机轴承三个损坏、伸展器限位损坏、手脚转动器手柄损坏、大转轮手柄损坏、脱漆太极揉推器手柄损坏	6
353	\N	\N	拼装地板	福城街道万地工业园南3栋宿舍楼下	陈挺	13751091955	\N	110平方	健身路径	\N	好家庭2022	1	8	2022-01-01	41	福城街道万地工业园南3栋宿舍楼下	[]	LH	4	四级压腿器、椭圆机、太极揉推器、上下肢训练器、太空漫步机、背肌训练器、上肢牵引器、钟摆器	上下肢训练器两边限位损坏	1
354	\N	\N	拼装地板	茜坑新村三区5栋	陈挺	13751091955	\N	110平方	健身路径	\N	杰威、澳瑞特、好家庭2017	1	6	2017年	41	茜坑新村三区5栋	[]	LH	4	压腿器、太极揉推器、腹肌板、象棋桌、腰背按摩器、三位扭腰器	腹肌板脱漆	1
355	\N	\N	\N	茜坑老村公园	陈挺	13751091955	\N	\N	室外智能健身器材	\N	\N	1	1	2021年	41	茜坑老村公园	[]	LH	4	\N	\N	0
356	\N	\N	\N	深圳市鑫金泉钻石刀具有限公司佰公坳工业区	陈挺	13751091955	\N	\N	篮球架（移动式）	\N	好家庭	1	2	2017-01-01	41	深圳市鑫金泉钻石刀具有限公司佰公坳工业区	[]	LH	4	两副篮球架	正常	0
357	\N	\N	拼装地板	招商锦绣观园6栋对面广场	陈挺	13751091955	\N	240平方	儿童游乐设施	\N	好家庭	1	1	2019-01-01	41	招商锦绣观园6栋对面广场	[]	LH	4	飞友牌儿童滑梯	玻璃罩缺失，拼装地板破损3平方	2
358	\N	\N	丙烯酸	亚翔精密塑胶五金（深圳）有限公司宿舍楼下篮球场	余志明	19925184310	\N	450平方	篮球架(固定式)	\N	杂牌	1	1	不详	42	亚翔精密塑胶五金（深圳）有限公司宿舍楼下篮球场	[]	LH	4	篮球架	篮板破损老旧，已达到报废；	1
359	\N	\N	安全地垫	田背一村37栋	余志明	19925184310	\N	100平方	健身路径	\N	好家庭2020	1	9	2020-01-01	42	田背一村37栋	[]	LH	4	告示牌x2、背肌训练器x2、椭圆机、上下肢训练器x2、上肢牵引器、太极揉推器、	椭圆机轴承损坏3个，太极揉推器转盘损坏一个、上肢牵引器拉绳缺失两条、两件下肢训练器限位损坏，地垫老旧破损	5
360	\N	\N	丙烯酸	田背一村14栋篮球场	余志明	19925184310	\N	450平方	篮球架（移动式）	\N	杂牌2020	1	1	2020-01-01	42	田背一村14栋篮球场	[]	LH	4	篮球架	两个篮网缺失	1
361	\N	\N	地砖	芷裕澜湾A、B栋架空层	余志明	19925184310	\N	65平方	健身路径	\N	杂牌	1	12	2013年	42	芷裕澜湾A、B栋架空层	[]	LH	4	告示牌x2，单杠x2，太空漫步机x3，跑步机，太极揉推器，健骑机，腰背按摩器，压腿器，	正常	0
362	\N	\N	拼装地板	楠木花园35栋	吴先生	18218178365	\N	120平方	健身路径	\N	杰威、澳瑞特、好家庭2022	1	13	2022-01-01	43	楠木花园35栋	[]	LH	4	太空漫步机、告示牌、围棋桌、腰背按摩器、手部腿部按摩器、腹肌板、三位扭腰器、上肢牵引器、腰背按摩器、单杠、伸背器、肋木架、、大转轮、	正常	0
363	\N	\N	拼装地板	迎侨花园小区花园7栋	吴先生	18218178365	\N	\N	儿童游乐设施	\N	好家庭2019	1	1	2019-01-01	43	迎侨花园小区花园7栋	[]	LH	4	飞友牌儿童滑梯	正常	0
364	/	\N	草地	金湖湾足球公园	刘小姐	13115268557	\N	/	足球场灯柱	\N	/	1	11	2022年	44	金湖湾足球公园	[]	LH	4	11杆灯柱	/	0
365	\N	\N	拼装地板	观湖街道祥澜苑1栋	沈娇露	18123765610	\N	50平方	健身路径	\N	好家庭	1	9	2013年	45	观湖街道祥澜苑1栋	[]	LH	5	腹肌板、伸背器、三位扭腰器，四级压腿器，太极揉推器，腰背按摩器，手部腿部按摩器，告示牌，象棋桌	正常	0
366	\N	\N	EPDM	观湖街道中航格澜花园一期9、10栋旁	沈娇露	18123765610	\N	90平方	健身路径	\N	桂宇星	1	15	2016年	45	观湖街道中航格澜花园一期9、10栋旁	[]	LH	5	三位扭腰器、上肢牵引器、大转轮、太极揉推器，压腿架，腹肌板x2、象棋桌、告示牌，健骑机，二位蹬力器、腰背按摩器、钟摆器、伸背器，太空漫步机、	颗粒地垫破损5平方	1
367	\N	\N	拼装地板	观湖街道中航格澜花园一期15栋旁	沈娇露	18123765610	\N	100平方	健身路径	\N	好家庭	1	12	2018年	45	观湖街道中航格澜花园一期15栋旁	[]	LH	5	告示牌、大转轮、手部腿部按摩器、腰背按摩器、太空漫步机、太极揉推器，三位扭腰器，四级压腿器，腹肌板、伸背器、健骑机，象棋桌、	正常	0
368	\N	\N	EPDM	观湖街道中航格澜花园二期17、15栋旁	沈娇露	18123765610	\N	130平方	健身路径	\N	长山城、杂牌	1	16	2009年	45	观湖街道中航格澜花园二期17、15栋旁	[]	LH	5	压腿器，腹肌板，太空漫步机x2，扭腰踏步组合x2、三位扭腰器，单人坐推训练器x2、跷跷板，太极揉推器，上肢牵引器，告示牌、二位蹬力器、钟摆器、天梯，	地面开裂、太空漫步机立柱松动，钟摆器摆腿缺失，两件单人坐推训练器推杆断缺报废，跷跷板盖帽缺失	5
369	\N	\N	安全地垫	观湖街道中航格澜花园二期6栋架空层	沈娇露	18123765610	\N	70平方	健身路径	\N	杂牌	1	8	2013年	45	观湖街道中航格澜花园二期6栋架空层	[]	LH	5	压腿架，伸背器，太极揉推器，腰背按摩器，上肢牵引器，三位扭腰器，太空漫步机，健骑机	太空漫步机两个摆腿缺失报废	1
370	\N	\N	EPDM55平方+草地36平方	观湖街道文澜苑5栋	沈娇露	18123765610	\N	EPDM55平方+草地36平方	健身路径	\N	澳瑞特、好家庭	1	14	2016年	45	观湖街道文澜苑5栋	[]	LH	5	告示牌、上肢牵引器、肋木架、二位蹬力器，太空漫步机、手部腿部按摩器，腰背按摩器，伸背器，四级压腿器，象棋桌，三位扭腰器，腹肌板，太极揉推器，健骑机	象棋桌座椅松动	1
371	\N	\N	EPDM	文澜苑5栋	沈娇露	18123765610	\N	60平方	儿童滑梯	\N	小博士	1	1	2018年	45	文澜苑5栋	[]	LH	5	儿童滑梯	单滑缺失，玻璃罩缺失，台阶缺失、地面磨损	4
372	\N	\N	拼装地板	观湖街道招商澜园北区1栋	沈娇露	18123765610	\N	60平方	健身路径	\N	好家庭	1	11	2016年	45	观湖街道招商澜园北区1栋	[]	LH	5	腹肌板、健骑机，伸背器，四级压腿器，三位扭腰器，太极揉推器，告示牌，太空漫步机，腰背按摩器，手部腿部按摩器，大转轮，	太空漫步机立柱松动	1
439	\N	\N	\N	新围第三工业区	何小姐	13691821215	\N	\N	篮球场	\N	未知	1	1	2009年	56	新围第三工业区	[]	LH	6	篮球场	正常	0
440	\N	\N	\N	新围新村	何小姐	13691821215	\N	\N	篮球场	\N	未知	1	1	2009年	56	新围新村	[]	LH	6	篮球场	正常	0
374	\N	\N	拼装地板	观湖街道东王新村20栋	胡欣秀	18822894603	\N	120平方	健身路径	\N	桂宇星	1	16	2014年	46	观湖街道东王新村20栋	[]	LH	5	告示牌、太空漫步机x2、三位扭腰器，上肢牵引器，压腿架，钟摆器、腹肌板x2，象棋桌，太极揉推器，二位蹬力器，伸背器、腰背按摩器、大转轮，健骑机	钟摆器踏板损坏变形，太极揉推器转盘轴承损坏，二位蹬力器两边限位损坏，健骑机限位损坏，大转轮缺失两边转轮	5
375	\N	\N	EPDM	观湖街道水晶山庄162栋	胡欣秀	18822894603	\N	100平方	健身路径	\N	杂牌	1	7	2013年	46	观湖街道水晶山庄162栋	[]	LH	5	单杠、三位扭腰器、太空漫步机x2、太极揉推器，腰背按摩器，双杠	太极揉推器盖帽缺失，三位扭腰器转盘缺失	2
376	\N	\N	硅pu	观湖街道水晶山庄106栋	胡欣秀	18822894603	\N	550平方	篮球场	\N	好家庭	1	1	2013年	46	观湖街道水晶山庄106栋	[]	LH	5	篮球场	篮网、防撞边条缺失，地层磨损	3
377	\N	\N	地砖	观湖街道观城工作站	胡欣秀	18822894603	\N	25平方	健身路径	\N	杂牌	1	5	2012年	46	观湖街道观城工作站	[]	LH	5	平步机、太空漫步机、腹肌板，三位扭腰器，太极揉推器	正常	0
378	\N	\N	水泥20平方+EPDM70平方	马坜老二村（马坜西区15号对面）	胡欣秀	18822894603	\N	水泥20平方+EPDM70平方	健身路径	\N	桂宇星	1	24	2012年	46	马坜老二村（马坜西区15号对面）	[]	LH	5	腹肌板x2、健骑机x2、太极揉推器x2、压腿架、伸背器x2、三位扭腰器x3、二位蹬力器x2、太空漫步机x2、腿部按摩器、上肢牵引器x2、腰背按摩器，钟摆器x2，大转轮，告示牌	epdm70平方破损，健骑机限位损坏x2，三位扭腰器转盘缺失，二位蹬力器限位损坏x2，大转轮手柄缺失，太空漫步机轴承损坏	6
379	\N	\N	水泥地	观湖街道岗头社区居委会	胡欣秀	18822894603	\N	100平方	健身路径	\N	澳瑞特	1	12	2013年	46	观湖街道岗头社区居委会	[]	LH	5	压腿器，三位扭腰器，肋木架，太空漫步机，柔韧训练器、上肢牵引器，腰背按摩器、手部腿部按摩器，太极揉推器，单杠、天梯、告示牌	太极揉推器轴承损坏，太空漫步机轴承损坏	2
380	\N	\N	地砖90平方+EPDM30平方	观城苑小区5、7栋	胡欣秀	18822894603	\N	地砖90平方+EPDM30平方	观健身路径	\N	杂牌	1	11	2022-11-29	46	观城苑小区5、7栋	[]	LH	5	太空漫步机x4、平步机x4、太极揉推器，健骑机x2，	正常	0
381	\N	\N	硅pu	观城苑小区7栋	胡欣秀	18822894603	\N	70平方	乒乓球台	\N	好家庭	1	4	2022-03-08	46	观城苑小区7栋	[]	LH	5	乒乓球台	地面面层磨损	1
382	\N	\N	EPDM	观湖街道伟禄雅苑1、2栋架空层	-	23737381	\N	270平方	健身路径	\N	好家庭	1	11	2013年	47	观湖街道伟禄雅苑1、2栋架空层	[]	LH	5	手部腿部按摩器、三位扭腰器、太极揉推器、上肢牵引器、双杠、健骑机，腰背按摩器，太空漫步机、告示牌、腹肌板	健骑机限位损坏，腰背按摩器滚轮缺失，太空漫步机摆腿缺失报废	3
383	\N	\N	EPDM	观湖街道观湖中心公园	-	23737381	\N	100平方	健身路径	\N	杂牌	1	8	2016年	47	观湖街道观湖中心公园	[]	LH	5	上肢牵引器，扭腰器，太空漫步机，二位蹬力器、坐腿坐拉，太极揉推器，直立健身车，蹬腿器	二位蹬力器限位损坏、坐腿坐拉限位损坏、扭腰器转盘损坏	3
384	\N	\N	拼装地板	观湖街道仁山智水F6栋一楼架空层	-	23737381	\N	30平方	健身路径	\N	好家庭	1	6	2013年	47	观湖街道仁山智水F6栋一楼架空层	[]	LH	5	手部腿部按摩器，伸背器，健骑机，三位扭腰器，四级压腿器，太极揉推器，	正常	0
385	\N	\N	拼装地板	观湖街道仁山智水F7栋一楼架空层	-	23737381	\N	40平方	健身路径	\N	好家庭	1	7	2014年	47	观湖街道仁山智水F7栋一楼架空层	[]	LH	5	腰背按摩器，上肢牵引器，太空漫步机，肋木架，二位蹬力器，腹肌板，象棋桌	二位蹬力器限位损坏	1
386	\N	\N	EPDM	观湖街道招商观园8栋	-	23737381	\N	100平方	健身路径	\N	长山城、好家庭	1	8	2019年	47	观湖街道招商观园8栋	[]	LH	5	双杠，腹肌板，伸背器，三位扭腰器，手部腿部按摩器，太极揉推器x2，大转轮	正常	0
387	\N	\N	拼装地板	观湖街道富士嘉园B座	-	23737381	\N	11平方	健身路径	\N	长山城、好家庭	1	17	2017年	47	观湖街道富士嘉园B座	[]	LH	5	太空漫步机x2，健骑机，四级压腿器，告示牌，腹肌板，三位扭腰器x2，手部腿部按摩器，太极揉推器x2，腰背按摩器x2，上肢牵引器，肋木架，压腿器，二位蹬力器	正常	0
388	\N	\N	/	高新科技园二路日海智能园三楼顶层	-	23737381	\N	\N	网球场	\N	/	1	1	2023年	47	高新科技园二路日海智能园三楼顶层	[]	LH	5	私人球场	/	0
389	\N	\N	丙烯酸	边防训练基地	陈思敏	13641446859	\N	1400平方	篮球场	\N	金陵	1	2	2022年	48	边防训练基地	[]	LH	5	篮球场	正常	0
390	\N	\N	拼装地板	观湖街道樟坑径上围村东区9号	陈思敏	13641446859	\N	100平方	健身路径	\N	长山城	1	16	2019年	48	观湖街道樟坑径上围村东区9号	[]	LH	5	腹肌板x2，太空漫步机x2，腰背按摩器x2，太极揉推器x2，压腿器x2，二位蹬力器x2，钟摆器x2，三位扭腰器x2，	正常	0
391	\N	\N	拼装地板	樟坑径党群服务中心	陈思敏	13641446859	\N	50平方	健身路径	\N	好家庭、澳瑞特2016	1	12	2016年	48	樟坑径党群服务中心	[]	LH	5	告示牌，平步机，二位蹬力器，象棋桌，太空漫步机，四季压腿器x2，腹肌板，腰背按摩器，三位扭腰器，太极揉推器，肋木架	平步机轴承损坏，二位蹬力器立柱螺丝松动，象棋桌三个座椅松动	3
392	\N	\N	安全地垫	观湖街道樟坑径公园	陈思敏	13641446859	\N	90平方	健身路径	\N	好家庭2016	1	8	2016年	48	观湖街道樟坑径公园	[]	LH	5	腹肌板，平步机，二位蹬力器，三位扭腰器，腰背按摩器，四级压腿器、肋木架x2，	正常	0
393	\N	\N	EPDM	观湖街道樟坑径下围社区公园	陈思敏	13641446859	\N	220平方	健身路径	\N	好家庭2017	1	12	2017年	48	观湖街道樟坑径下围社区公园	[]	LH	5	上肢牵引器，肋木架，腰背按摩器，手部腿部按摩器，太空漫步机，三位扭腰器，太极揉推器，四级压腿器，腹肌板，象棋桌，告示牌，健骑机，	健骑机限位损坏，三位扭腰器转盘缺失	2
394	\N	\N	水泥地、草地	观湖街道侨安科技工业园A栋	陈思敏	13641446859	\N	120平方	健身路径	\N	好家庭	1	12	2013年	48	观湖街道侨安科技工业园A栋	[]	LH	5	腹肌板，象棋桌，告示牌，平步机，三位扭腰器，四级压腿器，太空漫步机x2，腰背按摩器，二位蹬力器，肋木架	平步机轴承损坏。	1
441	\N	\N	\N	硕丰华工业区	何小姐	13691821215	\N	\N	篮球场	\N	未知	1	1	2009年	56	硕丰华工业区	[]	LH	6	篮球场	正常	0
395	\N	\N	拼装地板	观湖街道牛角龙工业园B2、B3、B4栋	陈思敏	13641446859	\N	180平方	健身路径	\N	长山城	1	24	2014年	48	观湖街道牛角龙工业园B2、B3、B4栋	[]	LH	5	腹肌板x2，平步机x2，三位扭腰器x2，钟摆器x2，上肢牵引器，腿部按摩器，大转轮，单人坐拉训练器，二位蹬力器x2，太空漫步机x2，压腿器，直立健身车，健骑机，棋牌桌，太极揉推器，告示牌，腰背按摩器，臂力训练器	平步机立柱断裂报废，钟摆器摆腿缺失报废，单人坐拉训练器螺丝缺失	3
396	\N	\N	拼装地板	观湖街道新田社区工作站	郑小姐	21089641	\N	130平方	健身路径	\N	好家庭2016	1	11	2016年	49	观湖街道新田社区工作站	[]	LH	5	告示牌，腹肌板，象棋桌，平步机，四级压腿器x2，太空漫步机，腰背按摩器，三位扭腰器，太极揉推器，二位蹬力器，	正常	0
397	\N	\N	拼装地板	新田公园东门	郑小姐	21089641	\N	65平方	健身路径	\N	澳瑞特2023	1	11	2023年	49	新田公园东门	[]	LH	5	三位扭腰器，健骑机，太极揉推器，伸背器，腰背按摩器，自重式下压训练器，告示牌，太空漫步机，上肢牵引器，多功能训练器，蹬力压腿训练器，	正常	0
398	\N	\N	拼装地板	新田名苑小区健身广场2栋	郑小姐	21089641	\N	130平方	健身路径	\N	澳瑞特2023	1	11	2023年	49	新田名苑小区健身广场2栋	[]	LH	5	三位扭腰器，健骑机，太极揉推器，伸背器，腰背按摩器，自重式下压训练器，告示牌，太空漫步机，上肢牵引器，多功能训练器，蹬力压腿训练器，	蹬力压腿训练器立柱松动，健骑机立柱松动	2
399	\N	\N	安全地垫	观湖街道古樟树公园西北门	郑小姐	21089641	\N	100平方	健身路径	\N	好家庭	1	12	2015年	49	观湖街道古樟树公园西北门	[]	LH	5	健骑机，上肢牵引器，伸背器x2，肋木架，腰背按摩器，三位扭腰器，太极揉推器，手部腿部按摩器，单杠，腹肌板，双杠	地垫老旧，健骑机限位损坏	2
400	\N	\N	地垫	观湖街道谷湖龙二村100栋	郑小姐	21089641	\N	80平方	健身路径	\N	好家庭	1	12	2013年	49	观湖街道谷湖龙二村100栋	[]	LH	5	上肢牵引器，单杠，二位蹬力器，手部腿部按摩器，四级压腿器，太极揉推器，伸背器，腰背按摩器，腹肌板，三位扭腰器，臂力训练器，晃板扭腰器	晃板扭腰器摆腿缺失，腰背按摩器滚轮损坏，手部腿部按摩器立柱松动	3
401	\N	\N	安全地垫	观湖街道君子嘉园C座	郑小姐	21089641	\N	90平方	健身路径	\N	杂牌	1	9	2010年	49	观湖街道君子嘉园C座	[]	LH	5	蹬力器，腹肌板，肋木架，双杠，三位扭腰器，椭圆机，太空漫步机，腰背按摩器，健身车	太空漫步机摆腿损坏断缺、轴承损坏；地垫破损老旧	3
402	\N	\N	EPDM	白鸽湖文化公园	卓小姐	13244853610	\N	220平方	力量1、常规1	\N	好家庭、杂牌、2019	1	17	2019年	50	白鸽湖文化公园	[]	LH	5	告示牌x2、坐式蹬力训练器、推举训练器、腿部训练器x2、手部训练器、扩胸训练器、大转轮、健骑机、手部腿部按摩器、三位扭腰器、太极揉推器、伸背器、四级压腿器，象棋桌、腹肌板	坐式蹬力训练器限位损坏、推举训练器限位损坏、腿部训练器限位损坏、象棋桌坐凳松动、手部腿部按摩器转盘缺失	5
403	\N	\N	安全地垫	观湖街道白鸽湖路95号	卓小姐	13244853610	\N	300平方	健身路径	\N	好家庭、杂牌、2015	1	25	2015年	50	观湖街道白鸽湖路95号	[]	LH	5	伸背器x2、平步机、健骑机x4、告示牌、四级压腿器、腰背按摩器、太空漫步机x2，腹肌板x2、手部腿部按摩器、三位扭腰器x2、太极揉推器x2、上肢牵引器x2、肋木架、单杠、双杠x2、	太空漫步机扶手脱出，健骑机限位损坏、两件上肢牵引器立柱松动，地垫老旧	4
404	\N	\N	安全地垫	观湖街道下围新花园A栋	卓小姐	13244853610	\N	200平方	健身路径	\N	桂宇星2019	1	21	2019年	50	观湖街道下围新花园A栋	[]	LH	5	上肢牵引器x2，腰背按摩器x2，太极揉推器x2，腿部按摩器x2，三位扭腰器x2，钟摆器、健骑机x2、太空漫步机x2，二位蹬力器，平步机x2，腹肌板x2，告示牌	两件健骑机轴承损坏，地垫老旧	2
405	\N	\N	安全地垫	观湖街道麒麟电子厂宿舍楼下	卓小姐	13244853610	\N	100平方	健身路径	\N	好家庭2019	1	12	2019年	50	观湖街道麒麟电子厂宿舍楼下	[]	LH	5	告示牌，伸背器，健骑机，腹肌板，太极揉推器，太空漫步机，三位扭腰器，手部腿部按摩器，伸背器，肋木架，上肢牵引器，腰背按摩器	健骑机立柱腐蚀报废，三位扭腰器转盘缺失三个，地垫老旧	3
406	\N	\N	\N	樟溪社区党群服务中心旁，深圳市龙华区观湖街道中森公园华府小区内	卓小姐	13244853610	\N	\N	室内乒乓球台	\N	杂牌	1	1	2022-12-09	50	樟溪社区党群服务中心旁，深圳市龙华区观湖街道中森公园华府小区内	[]	LH	5	乒乓球台	底座不稳、螺丝松动缺失，无法使用	1
407	\N	\N	EPDM	观湖街道大和村41栋	周牡	18825232251	\N	35平方	健身路径	\N	杂牌	1	8	2016年	51	观湖街道大和村41栋	[]	LH	5	悬空转轮、太空漫步机、钟摆器、腰背按摩器、太极揉推器、二位蹬力器、告示牌，上肢牵引器	太空漫步机三个轴承损坏，钟摆器两边轴承损坏，地垫破损35平方	3
408	\N	\N	EPDM	观湖街道环仔新村22栋	周牡	18825232251	\N	45平方	健身路径	\N	长山城	1	7	2014年	51	观湖街道环仔新村22栋	[]	LH	5	压腿器、腰背按摩器、二位蹬力器、三位扭腰器、大转轮、告示牌，太空漫步机	太空漫步机轴承损坏	1
409	\N	\N	拼装地板	观湖街道田寮新村48栋	周牡	18825232251	\N	55平方	健身路径	\N	好家庭	1	9	2016年	51	观湖街道田寮新村48栋	[]	LH	5	告示牌、太极揉推器、四级压腿器、腰背按摩器、钟摆器、二位蹬力器、腹肌板、太空漫步机、肋木架	正常	0
410	\N	\N	沥青	观湖街道田寮新村65栋	周牡	18825232251	\N	40平方	健身路径	\N	长山城	1	7	2013年	51	观湖街道田寮新村65栋	[]	LH	5	告示牌、压腿器、二位蹬力器、腰背按摩器，三位扭腰器、上肢牵引器、大转轮	大转轮手柄损坏，三位扭腰器转盘缺失，压腿器断缺一部分，	3
411	\N	\N	/	观湖街道新源社区老三村	余先生	18318871794	\N	15平方	健身路径	\N	杂牌	1	2	2008年	52	观湖街道新源社区老三村	[]	LH	5	蹬力器，双杠	正常	0
412	\N	\N	地砖	观湖街道新源社区老三村49栋	余先生	18318871794	\N	120平方	健身路径	\N	杂牌2019	1	12	2019年	52	观湖街道新源社区老三村49栋	[]	LH	5	钟摆器、三位扭腰器、太极揉推器，椭圆机，上肢牵引器，健骑机，腹肌板、平步机，腰背按摩器，划船器，太空漫步机，告示牌	划船器立柱断裂报废，腰背按摩器立柱松动，健骑机轴承损坏上肢牵引器边条脱出椭圆机轴承损坏，钟摆器轴承损坏	4
413	\N	\N	安全地垫	豪恩科技园	-	18816823069	\N	120平	健身路径	\N	体之杰	1	18	2006年	53	豪恩科技园	[]	LH	6	大转轮，双位腹肌板，棋牌桌*2，双人蹬力器，三人上肢牵引器，双杠，腰背按摩器，四级压腿按摩，三位扭腰器*2，引体向上训练器，太极揉推器，骑马器，展背器，钟摆器伸背器，单人漫步机，双位漫步机	钟摆器轴承损坏，腰背按摩器滚轮损坏，整体器材老旧	3
414	\N	\N	拼装地板	劳动者广场	-	18816823069	\N	100平	健身路径	\N	舒华	1	12	2015年	53	劳动者广场	[]	LH	6	蹬力器*2，展背器*2，太极揉推器，平步机，三位扭腰器，腹肌板，划船器，腰背按摩器，漫步机，告示牌	划船器护盖缺失，太极揉推器把手丢失*4，三位扭腰器轴承损坏，地面缺3平米	4
415	\N	\N	\N	万景工业园	-	18816823069	\N	500平	篮球场	\N	未知	1	1	2010年	53	万景工业园	[]	LH	6	篮球场	正常	0
416	\N	\N	\N	东一村	-	18816823069	\N	500平	篮球场	\N	未知	1	1	2013年	53	东一村	[]	LH	6	篮球场	正常	0
417	\N	\N	拼装地板	可乐园	-	18816823069	\N	120平	健身路径	\N	好家庭	1	11	2010年	53	可乐园	[]	LH	6	三位扭腰器，上肢牵引器，腰背按摩器，太极揉推器，蹬力器，多功能训练器，上下肢训练器，骑马器，太极揉推器，背腹训练器，告示牌	背腹训练器凳子损坏，上肢牵引器绳子缺失	2
418	\N	\N	安全地垫	万景工业区架空层	-	18816823069	\N	60平	健身路径	\N	奥瑞特	1	6	2017年	53	万景工业区架空层	[]	LH	6	三位扭腰器，上肢牵引器，腰背按摩器，太极揉推器，蹬力器，多功能训练器	三位扭腰器转盘轴承损坏	1
419	\N	\N	安全地垫	万景工业区边坡旁	-	18816823069	\N	20平	健身路径	\N	奥瑞特	1	3	2013年	53	万景工业区边坡旁	[]	LH	6	告示牌，二连单杠，天梯	地面破损20平	1
420	\N	\N	EPDM	下横朗新村2	-	18816823069	\N	100平	健身路径	\N	奥瑞特	1	13	2013年	53	下横朗新村2	[]	LH	6	三位扭腰器，棋牌桌*2，双位腹肌板，告示牌，太极揉推器，按摩揉推器，腰背按摩器，蹬力器，漫步机，弹振压腿器，健身车，秋千	腰背按摩器滚轮损坏，太极揉推器轴承损坏，秋千绳子缺失	3
421	\N	\N	拼装地板	桂冠华庭	-	18816823069	\N	80平	健身路径	\N	好家庭	1	12	2013年	53	桂冠华庭	[]	LH	6	腹肌板，棋牌桌，告示牌，展背器，太极揉推器，腰背按摩器，按摩揉推器，大转轮，三位扭腰器，漫步机，四级压腿按摩器，骑马器	正常	0
422	\N	\N	拼装地板	羊台山庄	-	18816823069	\N	100平	健身路径	\N	奥瑞特	1	11	2008年	53	羊台山庄	[]	LH	6	上肢牵引器，多功能锻炼器，腰背按摩器，转手器，展背器，太极揉推器，骑马器，自重式下压训练器，告示牌，三位扭腰器，漫步机	正常	0
423	\N	\N	水泥地	黄麻埔	-	21013182	\N	120平	健身路径	\N	奥瑞特	1	10	2005年	54	黄麻埔	[]	LH	6	斜躺健身车，肋木架，平步机，双位腹肌版，棋牌桌*2，三位扭腰器，展背器，按摩揉推器，太极揉推器，腰背按摩器，二连单杠，漫步机*2，仰卧起坐训练器，双杠，秋千，三人蹬力器，	三位扭腰器转盘轴承损坏，棋牌桌凳子缺*1，跷跷板损坏	3
424	\N	\N	硅PU	黄麻埔	-	21013182	\N	500平	篮球场	\N	奥瑞特	1	1	2005年	54	黄麻埔	[]	LH	6	篮球场	正常	0
425	\N	\N	安全地垫	同富邨1	-	21013182	\N	180平	健身路径	\N	舒华	1	14	2008年	54	同富邨1	[]	LH	6	双杠*2，上肢牵引器，双位扭腰器，蹬力器，腰背按摩器，漫步机，骑马器，三位扭腰器，双位腹肌板，弹振压腿器，棋牌桌，告示牌，双杠引体向上组合器	钟摆器摆臂缺失，钟摆器转盘轴承损坏，，漫步机轴承损坏，漫步机摆臂缺失，三位扭腰器转盘轴承损坏，地面破损8平米	6
426	\N	\N	硅PU	同富邨2	-	21013182	\N	500平	篮球场	\N	舒华	1	1	2008年	54	同富邨2	[]	LH	6	篮球场	正常	0
427	\N	\N	硅PU	上岭排	-	21013182	\N	500平	篮球场	\N	未知	1	1	2017年	54	上岭排	[]	LH	6	篮球场	正常	0
428	\N	\N	硅PU	下岭排	-	21013182	\N	500平	篮球场	\N	未知	1	1	2009年	54	下岭排	[]	LH	6	篮球场	正常	0
429	\N	\N	\N	大浪麒麟博物馆前社区公园	-	21013182	\N	未知	轨道式 中国象棋	\N	未知	1	4	2022-10-31	54	大浪麒麟博物馆前社区公园	[]	LH	6	中国象棋轨道式	凳子缺失*1，破损*4	2
430	\N	\N	拼装地板	大浪体育公园	-	13510681098	\N	640平	智能健身房	\N	好家庭	1	15	2022年	55	大浪体育公园	[]	LH	6	告示牌，棋牌桌，五边形体侧中心，腹背肌训练器，高啦推举训练器，腿部屈伸训练器，智能自发电手摇车，智能竞赛车*4，智能竞赛车（手摇式），战绳，三联单杠，双杠，肋木架，腹肌板，背肌训练器	腹背肌训练器屏幕失灵，智能竞赛车（手摇式）把手损坏	2
431	\N	\N	\N	大浪体育公园	-	13510681098	\N	500平	篮球场	\N	杂牌	1	1	2010年	55	大浪体育公园	[]	LH	6	篮球场	地面破损严重	1
432	\N	\N	水泥地	浪口一区	-	13510681098	\N	100平	健身路径	\N	杂牌	1	14	2010年	55	浪口一区	[]	LH	6	单人漫步机*2，双人扭腰器，太极揉推器，四级压腿按摩，三位扭腰器，大转轮，展背器，伸背器，双位腹肌板，双人坐拉器*2，平步机，倒立架	多件器材盖帽缺失*8，坐拉器拉手缺失*2，漫步机轴承损坏*1，漫步机摆腿缺失*1，倒立架损坏，三位扭腰器转盘损坏*1，缺失*1，双人扭腰器转盘损坏	8
433	\N	\N	\N	浪口一区	-	13510681098	\N	500平	篮球场	\N	杂牌	1	1	2010年	55	浪口一区	[]	LH	6	篮球场	正常	0
434	\N	\N	\N	美律电子有限公司	-	13510681098	\N	500平	篮球场	\N	杂牌	1	1	2010年	55	美律电子有限公司	[]	LH	6	篮球场	正常	0
435	\N	\N	EPDM	水围村富隆苑	-	13510681098	\N	30平	健身路径	\N	杂牌	1	7	不详	55	水围村富隆苑	[]	LH	6	双人骑马器，漫步机，坐拉器，三位扭腰器	物业自己采购，非管养	0
436	\N	\N	\N	华昌路239号B栋 博诚教育楼顶	-	13510681098	\N	500平	五人制笼式足球场	\N	杂牌	1	1	2020-12-21	55	华昌路239号B栋 博诚教育楼顶	[]	LH	6	足球场	正常	0
437	\N	\N	水泥地	颐丰华工业区2	何小姐	13691821215	\N	80平	健身路径	\N	奥瑞特	1	9	2012年	56	颐丰华工业区2	[]	LH	6	扭腰训练器，蹬力训练器，下拉训练器，前推训练器，多功能锻炼器，深蹲训练器，胸肌训练器，引体向上训练器，告示牌	胸肌训练器正常损坏*2	1
438	\N	\N	拼装地板	石凹办公楼	何小姐	13691821215	\N	120平	健身路径	\N	好家庭	1	3	2010年	56	石凹办公楼	[]	LH	6	腰背按摩器，按摩揉推器，二连单杠，	腰背按摩器滚轮损坏，地面破损15平米	2
442	\N	\N	拼装地板	华宁路91号14栋公园 华联丰大厦B1栋管露出前	何小姐	13691821215	\N	80平	健身路径	\N	奥瑞特	1	13	2023年	56	华宁路91号14栋公园 华联丰大厦B1栋管露出前	[]	LH	6	三位扭腰器，太极揉推器*3，展背器，自重式下压器，告示牌，漫步机，多功能锻炼器，上肢牵引器，腰背按摩器，蹬力压腿训练器，健骑机	正常	0
443	\N	\N	EPDM	水围新村	-	13728671543	\N	30平	健身路径	\N	杂牌	1	2	2005年	57	水围新村	[]	LH	6	三位扭腰器，太极揉推器	正常	0
444	\N	\N	EPDM	水围新村	-	13728671543	\N	500平	篮球场	\N	杂牌	1	1	不详	57	水围新村	[]	LH	6	篮球场	正常	0
445	\N	\N	EPDM	福龙家园	-	13728671543	\N	30平	健身路径	\N	杂牌	1	2	2013年	57	福龙家园	[]	LH	6	三位扭腰器，秋千	正常	0
446	\N	\N	水泥地	高峰苑山庄	-	13728671543	\N	40平	健身路径	\N	杂牌	1	5	2013年	57	高峰苑山庄	[]	LH	6	太极揉推器，漫步机，蹬力器，骑马器，三位扭腰器	整体器材老旧，三位扭腰器转盘损坏*3，	2
447	\N	\N	\N	高峰高架桥	-	13728671543	\N	500平	篮球场	\N	杂牌	1	1	不详	57	高峰高架桥	[]	LH	6	篮球场	正常	0
448	\N	\N	地砖	澳华花园	-	13728671543	\N	60平	健身路径	\N	杂牌	1	5	2013年	57	澳华花园	[]	LH	6	平步机，三位扭腰器，太极揉推器，漫步机，蹬力器	正常	0
449	\N	\N	拼装地板	龙胜公园	-	13713970552	\N	60平	健身路径	\N	好家庭	1	9	1999年	58	龙胜公园	[]	LH	6	腹肌板，骑马器，展背器，太极揉推器，腰背按摩器，按摩揉推器，漫步机，四级压腿按摩器，告示牌，	正常	0
450	\N	\N	\N	龙胜公园	-	13713970552	\N	\N	篮球场	\N	未知	1	1	不详	58	龙胜公园	[]	LH	6	篮球场	地面破损开裂，老旧	1
451	\N	\N	\N	龙胜社区工作站	-	13713970552	\N	\N	篮球场	\N	未知	1	1	不详	58	龙胜社区工作站	[]	LH	6	篮球场	地面破损开裂，老旧	1
452	\N	\N	EPDM	龙胜新村三区	-	13713970552	\N	100平	健身路径	\N	奥瑞特	1	13	2012年	58	龙胜新村三区	[]	LH	6	滑雪训练器*2，坐拉训练器*2，蹬力器*2，太极揉推器*2，三位扭腰器*2，漫步机*2，告示牌，	正常	0
453	\N	\N	EPDM	龙华路高峰学校宿舍楼斜对面小公园	-	13713970552	\N	60平	健身路径	\N	奥瑞特	1	11	2023年	58	龙华路高峰学校宿舍楼斜对面小公园	[]	LH	6	上肢牵引器，角力器，啊按摩揉推器，晃板扭腰器，腰背按摩器，腹肌板，太极揉推器，告示牌，三位扭腰器，漫步机，腰背按摩器，	地面整体破损60平，漫步机轴承损坏，腰背按摩器轴承损坏，晃板扭腰器倾斜	4
454	\N	\N	拼装地板	赖屋山东区42栋	-	13713970552	\N	500平	健身路径	\N	奥瑞特	1	11	2023年	58	赖屋山东区42栋	[]	LH	6	自重式下压训练器，上肢牵引器，健骑机，太极揉推器，腰背按摩器，展背器，告示牌，三位扭腰器，漫步机，多功能锻炼，蹬力压腿训练器	正常	0
455	\N	\N	地砖	和平里一期	/	-	\N	60平	健身路径	\N	奥瑞特	1	7	2018-01-01	59	和平里一期	[]	LH	6	压腿按摩器，跑步机，腰背按摩器，钟摆器，太极推手器，太空漫步机，太极揉推器	正常	0
456	\N	\N	安全地垫	和平里二期	/	-	\N	100平	健身路径	\N	奥瑞特	1	10	2018-01-01	59	和平里二期	[]	LH	6	天梯，蹬力器，腿部按摩器，腰背按摩器，三位扭腰器，健身车，腹肌板，平步机，太极揉推器，告示牌	正常	0
457	\N	\N	地砖/草坪	龙军花园	/	-	\N	500平	健身路径	\N	奥瑞特	1	27	2018-01-01	59	龙军花园	[]	LH	6	太空漫步机，象棋桌，腰背按摩器，单杠，三位扭腰器，太极揉推器，腰背按摩器，太空漫步机，按摩揉推器，太极推手器，骑马器，四级压腿器，伸背器，告示牌，大转轮，腰背按摩器，三位扭腰器，腹肌板，双杠，三位扭腰器，天梯，腰背按摩器，太极揉推器，腰背按摩器，腹肌板，三位扭腰器，太空漫步机	正常	0
458	\N	\N	安全地垫	鹏华香域花园	/	-	\N	100平	健身路径	\N	奥瑞特	1	9	2018-01-01	59	鹏华香域花园	[]	LH	6	压腿按摩器，骑马器，太空漫步机，天梯，钟摆器，骑马器，跷跷板，太空漫步机，太极推手器	正常	0
459	\N	\N	水泥地	光浩工业园	/	-	\N	500平	篮球场	\N	\N	1	1	未知	60	光浩工业园	[]	LH	6	篮球场	\N	0
1	\N	\N	硅pu	水斗富豪新村	钟家燕	13420955222	\N	70	健身路径	\N	好家庭	1	14	2016年	1	水斗富豪新村	["http://localhost:8000/static/uploads/inspection/2026/04/site_overall_07022858_34207603.jpg"]	LH	1	蹬力器，腰背按摩器，按摩揉推器，上肢牵引器，三位扭腰器，肋木架，骑马器，太极推手器，太空漫步机，四级压腿按摩器，腹肌训练板，伸背器，棋牌桌，告示牌	拼装地板缺17㎡，象棋桌松动，骑马器限位损坏，蹬力器限位损坏，太空漫步机补漆	5
\.


--
-- Name: community_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.community_id_seq', 60, true);


--
-- Name: dictionary_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.dictionary_id_seq', 24, true);


--
-- Name: equipment_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.equipment_id_seq', 4, true);


--
-- Name: inspection_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.inspection_id_seq', 181, true);


--
-- Name: maintenance_history_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.maintenance_history_id_seq', 1, false);


--
-- Name: maintenance_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.maintenance_id_seq', 7, true);


--
-- Name: street_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.street_id_seq', 1, false);


--
-- Name: user_data_permissions_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_data_permissions_id_seq', 1, false);


--
-- Name: user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.user_id_seq', 7, true);


--
-- Name: venue_id_seq; Type: SEQUENCE SET; Schema: public; Owner: postgres
--

SELECT pg_catalog.setval('public.venue_id_seq', 459, true);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: community community_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.community
    ADD CONSTRAINT community_pkey PRIMARY KEY (id);


--
-- Name: dictionary dictionary_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dictionary
    ADD CONSTRAINT dictionary_pkey PRIMARY KEY (id);


--
-- Name: equipment equipment_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.equipment
    ADD CONSTRAINT equipment_pkey PRIMARY KEY (id);


--
-- Name: inspection inspection_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.inspection
    ADD CONSTRAINT inspection_pkey PRIMARY KEY (id);


--
-- Name: maintenance_history maintenance_history_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance_history
    ADD CONSTRAINT maintenance_history_pkey PRIMARY KEY (id);


--
-- Name: maintenance maintenance_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance
    ADD CONSTRAINT maintenance_pkey PRIMARY KEY (id);


--
-- Name: street street_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.street
    ADD CONSTRAINT street_pkey PRIMARY KEY (id);


--
-- Name: user_data_permissions user_data_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_data_permissions
    ADD CONSTRAINT user_data_permissions_pkey PRIMARY KEY (id);


--
-- Name: user user_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT user_pkey PRIMARY KEY (id);


--
-- Name: venue venue_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.venue
    ADD CONSTRAINT venue_pkey PRIMARY KEY (id);


--
-- Name: ix_community_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_community_name ON public.community USING btree (name);


--
-- Name: ix_dictionary_dict_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_dictionary_dict_code ON public.dictionary USING btree (dict_code);


--
-- Name: ix_equipment_category; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_equipment_category ON public.equipment USING btree (category);


--
-- Name: ix_equipment_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_equipment_name ON public.equipment USING btree (name);


--
-- Name: ix_street_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_street_name ON public.street USING btree (name);


--
-- Name: ix_user_data_permissions_user_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_data_permissions_user_id ON public.user_data_permissions USING btree (user_id);


--
-- Name: ix_user_openid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_user_openid ON public."user" USING btree (openid);


--
-- Name: ix_user_phone; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_user_phone ON public."user" USING btree (phone);


--
-- Name: community community_street_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.community
    ADD CONSTRAINT community_street_id_fkey FOREIGN KEY (street_id) REFERENCES public.street(id);


--
-- Name: inspection inspection_venue_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.inspection
    ADD CONSTRAINT inspection_venue_id_fkey FOREIGN KEY (venue_id) REFERENCES public.venue(id);


--
-- Name: maintenance maintenance_inspection_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance
    ADD CONSTRAINT maintenance_inspection_id_fkey FOREIGN KEY (inspection_id) REFERENCES public.inspection(id);


--
-- Name: maintenance maintenance_venue_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maintenance
    ADD CONSTRAINT maintenance_venue_id_fkey FOREIGN KEY (venue_id) REFERENCES public.venue(id);


--
-- Name: user_data_permissions user_data_permissions_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.user_data_permissions
    ADD CONSTRAINT user_data_permissions_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);


--
-- Name: venue venue_community_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.venue
    ADD CONSTRAINT venue_community_id_fkey FOREIGN KEY (community_id) REFERENCES public.community(id);


--
-- PostgreSQL database dump complete
--

\unrestrict LKO09nzkYpC3n8yYhYCbY0Ub2aR8ZJj4eQ3BAIp6W5HpHC1iE7mK6EKFcQbQtdX

--
-- Database "postgres" dump
--

--
-- PostgreSQL database dump
--

\restrict zd7cUWUitqbsvMaa5BXT7gB5SxTQ8HjUx8u0h2xv4dHGIrn4ieeUxZWzDHT0Vqg

-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DROP DATABASE postgres;
--
-- Name: postgres; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE postgres WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'en_US.utf8';


ALTER DATABASE postgres OWNER TO postgres;

\unrestrict zd7cUWUitqbsvMaa5BXT7gB5SxTQ8HjUx8u0h2xv4dHGIrn4ieeUxZWzDHT0Vqg
\connect postgres
\restrict zd7cUWUitqbsvMaa5BXT7gB5SxTQ8HjUx8u0h2xv4dHGIrn4ieeUxZWzDHT0Vqg

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: DATABASE postgres; Type: COMMENT; Schema: -; Owner: postgres
--

COMMENT ON DATABASE postgres IS 'default administrative connection database';


--
-- PostgreSQL database dump complete
--

\unrestrict zd7cUWUitqbsvMaa5BXT7gB5SxTQ8HjUx8u0h2xv4dHGIrn4ieeUxZWzDHT0Vqg

--
-- PostgreSQL database cluster dump complete
--

