import json
import argparse
import openai
import time
from tqdm import tqdm
from collections import Counter
import sqlite3
import glob
import os
import csv
import io

openai.api_key = ""

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--output_path", type=str)

    opt = parser.parse_args()

    return opt

def make_prompt(question, schema, schema_description):
    prompt = """Your role is an evidence generator to assist with text-to-SQL operations. You must create the external knowledge for text-to-SQL operations as evidence.
    
Purpose: Create an evidence to aid text-to-SQL tasks
action
  1. Please refer to the given question, evidence, and SQL pair and the DB Schema of samples, and schema description.
  2. For the given question, schema and schema description, generate evidence in one sentence to help text-to-sql.
  3. The length of the evidence should be as short as possible and does not need to be generated if unnecessary.
  4. Skip the description and just print out evidence.

### samples ####################################################
1. Question, evidence, and SQL pair samples
{
    {
        "question": "Which date has the most ordered quantity? What is the total order quantity on that day?",
        "evidence": "total quantity refers to qty; most ordered quantity refers to order with the highest quantity where MAX(sum(qty))",
        "SQL": "SELECT ord_date, SUM(qty) FROM sales GROUP BY ord_date ORDER BY SUM(qty) DESC LIMIT 1"
    },
    {
        "question": "What is the title with the most ordered quantity in year 1992?",
        "evidence": "total quantity refers to qty; most ordered quantity refers to order with the highest quantity where MAX(count(qty)); date refers to ord_date; year 1992 refers to YEAR(ord_date) = 1992",
        "SQL": "SELECT T2.title FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y', T1.ord_date) = '1992' ORDER BY T1.qty DESC LIMIT 1"
    },
    {
        "question": "List the title, price and publication date for all sales with 'ON invoice' payment terms.",
        "evidence": "publication date refers to pubdate; payment terms refers to payterms; payterms = 'ON invoice'",
        "SQL": "SELECT T2.title, T2.price, T2.pubdate FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id WHERE T1.payterms = 'ON invoice'"
    },
    {
        "question": "What is the title that have at least 10% royalty without minimum range amount.",
        "evidence": "at least 10% royalty refers to royalty > = 10; minimum range is synonym for low range which refers to lorange; without minimum range amount refers to lorange <> 0",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange = 0 AND T2.royalty >= 10"
    },
    {
        "question": "State the title and royalty percentage for title ID BU2075 between 10000 to 50000 range.",
        "evidence": "lorange mean low range; hirange mean high range; range refers to between the low and high range; lorange>10000; hirange<12000",
        "SQL": "SELECT T1.title, T2.royalty FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange > 10000 AND T2.hirange < 50000 AND T1.title_ID = 'BU2075'"
    },
    {
        "question": "Among the titles with royalty percentage, which title has the greatest royalty percentage. State it's minimum range to enjoy this royalty percentage.",
        "evidence": "minimum range is synonym for low range which refers to lorange",
        "SQL": "SELECT T1.title, T2.lorange FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id ORDER BY T2.royalty DESC LIMIT 1"
    },
    {
        "question": "Provide a list of titles together with its publisher name for all publishers located in the USA.",
        "evidence": "publisher name refers to pub_name;",
        "SQL": "SELECT T1.title, T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = 'USA'"
    },
    {
        "question": "State the royalty percentage for the most year to date sale title within the 20000 range.",
        "evidence": "most year to date sales refers to MAX(ytd_sales); range limit means high range which refers to hirange; the 20000 range refers to hirange<20000",
        "SQL": "SELECT MAX(T1.ytd_sales) FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.lorange > 20000 AND T2.hirange < 20000"
    },
    {
        "question": "List all titles published in year 1991. Also provide notes details of the title and the publisher's name.",
        "evidence": "publisher name refers to pub_name; publication date refers to pubdate; published in year 1991 refers to YEAR(pubdate) = 1991",
        "SQL": "SELECT T1.title, T1.notes, T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE STRFTIME('%Y', T1.pubdate) = '1991'"
    },
    {
        "question": "List all titles with sales of quantity more than 20 and store located in the CA state.",
        "evidence": "qty is abbreviation for quantity; sales of quantity more than 20 refers to qty>20; store refers to stor_name",
        "SQL": "SELECT T1.title, T2.qty FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id INNER JOIN stores AS T3 ON T2.stor_id = T3.stor_id WHERE T2.qty > 20 AND T3.state = 'CA'"
    },
    {
        "question": "Name the store with the highest quantity in sales? What is the least quantity title from the store's sale?",
        "evidence": "qty is abbreviation for quantity; highest quantity refers to MAX(qty); least quantity refers to MIN(qty)",
        "SQL": "SELECT T3.stor_id, T2.title FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id INNER JOIN stores AS T3 ON T3.stor_id = T1.stor_id WHERE T3.stor_id = ( SELECT stor_id FROM sales GROUP BY stor_id ORDER BY SUM(qty) DESC LIMIT 1 ) GROUP BY T3.stor_id, T2.title ORDER BY SUM(T1.qty) ASC LIMIT 1"
    },
    {
        "question": "Name the title and publisher for title ID BU 2075. Provide all the royalty percentage for all ranges.",
        "evidence": "name the publisher refers to pub_name",
        "SQL": "SELECT T1.title, T3.pub_name, T2.lorange, T2.hirange, T2.royalty FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id INNER JOIN publishers AS T3 ON T1.pub_id = T3.pub_id WHERE T1.title_id = 'BU2075'"
    },
    {
        "question": "Name the store with ID 7066 and calculate the percentage of the the quantity ordered that were on 'Net 30' payment terms.",
        "evidence": "store with ID 7066 refers to stor_ID = '7066'; 'Net 60' payment terms refers to payterm = 'Net 60'; qty is abbreviation for quantity; percentage = DIVIDE(payterms = 'Net 60', sum(qty))*100",
        "SQL": "SELECT T2.stor_name , CAST(SUM(CASE WHEN payterms = 'Net 30' THEN qty ELSE 0 END) AS REAL) * 100 / SUM(qty) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE T1.stor_id = '7066' GROUP BY T2.stor_name"
    },
    {
        "question": "State the publisher name for publisher ID 877? Calculate its average year to date sales.",
        "evidence": "publisher id refers to pub_id; publisher name refers to pub_name; average year to date sales = AVG(ytd_sales)",
        "SQL": "SELECT T2.pub_name, AVG(T1.ytd_sales) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.pub_id = '0877' GROUP BY T2.pub_name"
    },
    {
        "question": "Name all employees who were hired before year 1990.",
        "evidence": "hired before year 1990 refers to YEAR(hire_date)<1990",
        "SQL": "SELECT fname, lname FROM employee WHERE STRFTIME('%Y', hire_date) < '1990'"
    },
    {
        "question": "Which employee has the lowest job level. State the first name, last name and when he /she was hired.",
        "evidence": "lowest job level refers to MIN(job_lvl)",
        "SQL": "SELECT fname, lname, hire_date FROM employee ORDER BY job_lvl LIMIT 1"
    },
    {
        "question": "In which year has the most hired employees?",
        "evidence": "most hired employees refers to MAX(count(emp_id))",
        "SQL": "SELECT STRFTIME('%Y', hire_date) FROM employee GROUP BY STRFTIME('%Y', hire_date) ORDER BY COUNT(emp_id) DESC LIMIT 1"
    },
    {
        "question": "List all employees who are at the maximum level in their job designation.",
        "evidence": "maximum level in their job designation refers to job_lvl = MAX(max_lvl)",
        "SQL": "SELECT T1.fname, T1.lname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.job_lvl = T2.max_lvl"
    },
    {
        "question": "Name the Chief Executive Officer and when he/she was hired.",
        "evidence": "Chief Financial Offer is a job description which refers to job_desc",
        "SQL": "SELECT T1.fname, T1.lname, T1.hire_date FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = 'Chief Financial Officier'"
    },
    {
        "question": "Who are the employees working for publisher not located in USA? State the employee's name and publisher name.",
        "evidence": "not located at USA refers to country! = 'USA'",
        "SQL": "SELECT T1.fname, T1.lname, T2.pub_name FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country != 'USA'"
    },
    {
        "question": "List all employees working for publisher 'GGG&G'. State their name and job description.",
        "evidence": "name = fname, lname; job description refers to job_desc; publisher refers pub_name",
        "SQL": "SELECT T1.fname, T1.lname, T3.job_desc FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN jobs AS T3 ON T1.job_id = T3.job_id WHERE T2.pub_name = 'GGG&G'"
    },
    {
        "question": "For each publisher, state the type of titles they published order by the publisher name.",
        "evidence": "publisher name refers to pub_name",
        "SQL": "SELECT DISTINCT T2.pub_name, T1.type FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id ORDER BY T2.pub_name"
    },
    {
        "question": "Name the publisher which has the most titles published in 1991.",
        "evidence": "most title published refers to MAX(count(title_id); published in 1991 refers to YEAR(pubdate) = 1991",
        "SQL": "SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE STRFTIME('%Y', T1.pubdate) = '1991' GROUP BY T1.pub_id, T2.pub_name ORDER BY COUNT(T1.title_id) DESC LIMIT 1"
    },
    {
        "question": "Name the title with the highest price published by 'Binnet & Hardley'.",
        "evidence": "published by refers to pub_name",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = 'Binnet & Hardley' ORDER BY T1.price DESC LIMIT 1"
    },
    {
        "question": "Among all employees, who have job level greater than 200. State the employee name and job description.",
        "evidence": "job level greater than 200 refers to job_lvl>200; job description refers to job_desc",
        "SQL": "SELECT T1.fname, T1.lname, T2.job_desc FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.job_lvl > 200"
    },
    {
        "question": "Name all the authors for all business titles.",
        "evidence": "business title refers to title under business where type = 'business'",
        "SQL": "SELECT T3.au_fname, T3.au_lname FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T1.type = 'business'"
    },
    {
        "question": "List all the titles and year to date sales by author who are not on contract.",
        "evidence": "year to date sales refers to ytd_sales; not on contract refers to contract = 0",
        "SQL": "SELECT T1.title_id, T1.ytd_sales FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T3.contract = 0"
    },
    {
        "question": "For all authors from CA who are not on contract, which title of his/hers has the most year to date sales.",
        "evidence": "year to date sales refers to ytd_sales; on contract refers to contract = 1",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T3.contract = 0 AND T3.state = 'CA' ORDER BY T1.ytd_sales DESC LIMIT 1"
    },
    {
        "question": "Name all the authors for 'Sushi, Anyone?'.",
        "evidence": "most year to date sales refers to MAX(ytd_sales); on contract refers to contract = 1; name of author = au_fname, au_lname",
        "SQL": "SELECT T3.au_fname, T3.au_lname FROM titles AS T1 INNER JOIN titleauthor AS T2 ON T1.title_id = T2.title_id INNER JOIN authors AS T3 ON T2.au_id = T3.au_id WHERE T1.title = 'Sushi, Anyone?'"
    },
    {
        "question": "Calculate the percentage of the employees who are Editor or Designer?",
        "evidence": "Editor or Auditor are job description which refers to job_desc; percentage = DIVIDE(count(job_desc = 'Editor' or job_desc = 'Auditor'), count(emp_id))*100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.job_desc IN ('Editor', 'Designer') THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id"
    },
    {
        "question": "List all titles which have year to date sales higher than the average order by pubisher name.",
        "evidence": "year to date sales refers to ytd_sales; average order = AVG(ytd_sales)",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.ytd_sales > ( SELECT AVG(ytd_sales) FROM titles )"
    },
    {
        "question": "How many publishers are in the USA?",
        "evidence": "",
        "SQL": "SELECT COUNT(pub_id) FROM publishers WHERE country = 'USA'"
    },
    {
        "question": "What is the publisher's information of New Moon Books?",
        "evidence": "publisher name refers to pub_name; New Moon Books is a publisher name",
        "SQL": "SELECT T1.pr_info FROM pub_info AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = 'New Moon Books'"
    },
    {
        "question": "Please list the first names of the employees who work as Managing Editor.",
        "evidence": "Managing Editor is a job description which refers to job_desc",
        "SQL": "SELECT T1.fname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T2.job_desc = 'Managing Editor'"
    },
    {
        "question": "What is the highest level of job to get to for the employee who got hired the earliest?",
        "evidence": "highest job level refers to MAX(job_lvl); hired the earliest refers to MIN(hire_date)",
        "SQL": "SELECT T2.max_lvl FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id ORDER BY T1.hire_date LIMIT 1"
    },
    {
        "question": "In which city is the store with the highest total sales quantity located?",
        "evidence": "qty is abbreviation for quantity; highest sales quantity refers to MAX(qty)",
        "SQL": "SELECT T2.city FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id GROUP BY T2.city ORDER BY SUM(T1.qty) DESC LIMIT 1"
    },
    {
        "question": "What is the price of the book that sells the best?",
        "evidence": "qty is abbreviation for quantity; sells the best mean with the most sales quantity; MAX(qty)",
        "SQL": "SELECT T2.price FROM sales AS T1 INNER JOIN titles AS T2 ON T1.title_id = T2.title_id ORDER BY T1.qty DESC LIMIT 1"
    },
    {
        "question": "Please list the stores that ordered the book \"Life Without Fear\".",
        "evidence": "store name refers to stor_name",
        "SQL": "SELECT T2.stor_name FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T3.title = 'Life Without Fear'"
    },
    {
        "question": "Among the stores that have ordered the book \"Life Without Fear\", how many of them are located in Massachusetts?",
        "evidence": "Massachusetts is a state",
        "SQL": "SELECT COUNT(T1.stor_id) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T2.state = 'Massachusetts'"
    },
    {
        "question": "In which country is the publisher of the book \"Life Without Fear\" located?",
        "evidence": "Life Without Fear is book title",
        "SQL": "SELECT T2.country FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.title = 'Life Without Fear'"
    },
    {
        "question": "What is the publisher that has published the most expensive book?",
        "evidence": "most expensive book refers to MAX(price)",
        "SQL": "SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id ORDER BY T1.price DESC LIMIT 1"
    },
    {
        "question": "Among the publishers in the USA, how many of them have published books that are over $15?",
        "evidence": "are over $15 refers to price>15",
        "SQL": "SELECT COUNT(DISTINCT T1.pub_id) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = 'USA' AND T1.price > 15"
    },
    {
        "question": "Please give more detailed information about the first three books that sell the best.",
        "evidence": "qty is abbreviation for quantity; sells the best mean with the most sales quantity; MAX(qty)",
        "SQL": "SELECT T1.notes FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id ORDER BY T2.qty DESC LIMIT 3"
    },
    {
        "question": "How many books on business have the bookstores in Massachusetts ordered?",
        "evidence": "Massachusetts is a state; business books refers to type = 'business'",
        "SQL": "SELECT SUM(T1.qty) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id INNER JOIN titles AS T3 ON T1.title_id = T3.title_id WHERE T2.state = 'Massachusetts' AND T3.type = 'business'"
    },
    {
        "question": "What is the average quantity of each order for the book \"Life Without Fear\"?",
        "evidence": "qty is abbreviation for quantity; average quantity order = AVG(qty)",
        "SQL": "SELECT CAST(SUM(T2.qty) AS REAL) / COUNT(T1.title_id) FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE T1.title = 'Life Without Fear'"
    },
    {
        "question": "What is the average level employees working as Managing Editor are at? How many levels are there between the average level and the highest level?",
        "evidence": "Managing Editor is a job description which refers to job_desc; job level refers to job_lvl; highest level job refers to max_lvl; levels between the average level and the highest level = SUBTRACT(max_lvl; AVG(job_lvl))",
        "SQL": "SELECT AVG(T2.job_lvl), T1.max_lvl - AVG(T2.job_lvl) FROM jobs AS T1 INNER JOIN employee AS T2 ON T1.job_id = T2.job_id WHERE T1.job_desc = 'Managing Editor' GROUP BY T2.job_id, T1.max_lvl"
    },
    {
        "question": "Which one is the cheapest business book?",
        "evidence": "business books refers to type = 'business'; cheapest book refers to MIN(price)",
        "SQL": "SELECT title FROM titles WHERE type = 'business' ORDER BY price LIMIT 1"
    },
    {
        "question": "Which type of book had the most pre-paid amount?",
        "evidence": "most pre-paid amount refers to MAX(advance)",
        "SQL": "SELECT type FROM titles ORDER BY advance DESC LIMIT 1"
    },
    {
        "question": "What's the royalty for the bestseller book?",
        "evidence": "qty is abbreviation for quantity; bestseller means with the most sales quantity; MAX(qty)",
        "SQL": "SELECT royalty FROM titles ORDER BY ytd_sales DESC LIMIT 1"
    },
    {
        "question": "Which job level is O'Rourke at?",
        "evidence": "job level refers to job_lvl",
        "SQL": "SELECT job_lvl FROM employee WHERE lname = 'O''Rourke'"
    },
    {
        "question": "Show me the employ id of the highest employee who doesn't have a middle name.",
        "evidence": "highest employee refers to employee with the highest job level; MAX(job_lvl)",
        "SQL": "SELECT emp_id FROM employee WHERE minit = '' ORDER BY job_lvl DESC LIMIT 1"
    },
    {
        "question": "Is the author of \"Sushi, Anyone?\" on the contract?",
        "evidence": "contract = 1 means on contract; contract = 0 means not on contract",
        "SQL": "SELECT T1.contract FROM authors AS T1 INNER JOIN titleauthor AS T2 ON T1.au_id = T2.au_id INNER JOIN titles AS T3 ON T2.title_id = T3.title_id WHERE T3.title = 'Sushi, Anyone?'"
    },
    {
        "question": "Which publisher had the highest job level? Give his/her full name.",
        "evidence": "highest job level refers to MAX(job_lvl)",
        "SQL": "SELECT T1.fname, T1.minit, T1.lname FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id ORDER BY T1.job_lvl DESC LIMIT 1"
    },
    {
        "question": "What's Pedro S Afonso's job title?",
        "evidence": "job title means job description which refers to job_desc",
        "SQL": "SELECT T2.job_desc FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.fname = 'Pedro' AND T1.minit = 'S' AND T1.lname = 'Afonso'"
    },
    {
        "question": "How many levels are there left for Diego W Roel to reach if he/she could go to the max level for his/her position?",
        "evidence": "max level for his position refers to max_lvl; job level refers to job_lvl; level left to reach the max = SUBTRACT(max_lvl, job_lvl)",
        "SQL": "SELECT T2.max_lvl - T1.job_lvl FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id WHERE T1.fname = 'Diego' AND T1.minit = 'W' AND T1.lname = 'Roel'"
    },
    {
        "question": "What's on the notes for the order happened on 1994/9/14?",
        "evidence": "order happened on refers to ord_date",
        "SQL": "SELECT T1.notes FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y-%m-%d', T2.ord_date) = '1994-09-14'"
    },
    {
        "question": "List the type of the book for the order which was sold on 1993/5/29.",
        "evidence": "sold on refers to ord_date",
        "SQL": "SELECT DISTINCT T1.type FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE STRFTIME('%Y-%m-%d', T2.ord_date) = '1993-05-29'"
    },
    {
        "question": "Tell me about the information of the French publisher.",
        "evidence": "French publisher means publisher in France where country = 'France'",
        "SQL": "SELECT T1.pr_info FROM pub_info AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.country = 'France'"
    },
    {
        "question": "What's the publisher of the book \"Silicon Valley Gastronomic Treats\"? Give the publisher's name.",
        "evidence": "publisher name refers to pub_name; Silicon Valley Gastronomic Treats is the title of a book",
        "SQL": "SELECT T2.pub_name FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.title = 'Silicon Valley Gastronomic Treats'"
    },
    {
        "question": "Which city did Victoria P Ashworth work in?",
        "evidence": "",
        "SQL": "SELECT T2.city FROM employee AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.fname = 'Victoria' AND T1.minit = 'P' AND T1.lname = 'Ashworth'"
    },
    {
        "question": "How many sales did the store in Remulade make?",
        "evidence": "Remulade is a city; sales in the store refers to ord_num",
        "SQL": "SELECT COUNT(T1.ord_num) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE T2.city = 'Remulade'"
    },
    {
        "question": "For the quantities, what percent more did the store in Fremont sell than the store in Portland in 1993?",
        "evidence": "qty is abbreviation for quantity; Fremont and Portland are name of city; sell in 1993 refers to YEAR(ord_date) = 1993; percentage = DIVIDE(\nSUBTRACT(SUM(qty where city = \u2018Fremont\u2019 and year(ord_date = 1993)), \nSUM(qty where city = \u2018Portland\u2019 and year(ord_date = 1993))), SUM(qty where city = \u2018Fremont\u2019 and year(ord_date = 1993)) *100",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.city = 'Fremont' THEN qty END) - SUM(CASE WHEN T2.city = 'Portland' THEN qty END) AS REAL) * 100 / SUM(CASE WHEN T2.city = 'Fremont' THEN qty END) FROM sales AS T1 INNER JOIN stores AS T2 ON T1.stor_id = T2.stor_id WHERE STRFTIME('%Y', T1.ord_date) = '1993'"
    },
    {
        "question": "Among all the employees, how many percent more for the publishers than designers?",
        "evidence": "publisher and designer are job descriptions which refers to job_desc; percentage more = 100*(SUBTRACT(SUM(CASE WHERE job_desc = 'publisher), SUM(CASE WHERE job_desc = 'designer'))",
        "SQL": "SELECT CAST(SUM(CASE WHEN T2.job_desc = 'publisher' THEN 1 ELSE 0 END) - SUM(CASE WHEN T2.job_desc = 'designer' THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.job_id) FROM employee AS T1 INNER JOIN jobs AS T2 ON T1.job_id = T2.job_id"
    },
    {
        "question": "Find and list the full name of employees who were hired between 1990 and 1995. Also, arrange them in the descending order of job level.",
        "evidence": "job level refers to job_lvl; YEAR(hire_date) between 1990 and 1995",
        "SQL": "SELECT fname, minit, lname FROM employee WHERE STRFTIME('%Y', hire_date) BETWEEN '1990' AND '1995' ORDER BY job_lvl DESC"
    },
    {
        "question": "Which titles has above average royalty rate? Give those title's name, type and price?",
        "evidence": "average royalty rate = DIVIDE(SUM(royalty), COUNT(title_id))",
        "SQL": "SELECT DISTINCT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN roysched AS T2 ON T1.title_id = T2.title_id WHERE T2.royalty > ( SELECT CAST(SUM(royalty) AS REAL) / COUNT(title_id) FROM roysched )"
    },
    {
        "question": "In 1994 which title had less order quanty than the average order quantity? Find the title name, type and price.",
        "evidence": "orders in 1994 refers to YEAR(ord_date) = 1994; order quantity refers to number of order expressed by ord_num; average order quantity = DIVIDE(SUM(ord_num), COUNT(title_id))",
        "SQL": "SELECT DISTINCT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id WHERE T2.ord_date LIKE '1994%' AND T2.Qty < ( SELECT CAST(SUM(T4.qty) AS REAL) / COUNT(T3.title_id) FROM titles AS T3 INNER JOIN sales AS T4 ON T3.title_id = T4.title_id )"
    },
    {
        "question": "List the title name, type, and price of the titles published by New Moon Books. Arrange the list in ascending order of price.",
        "evidence": "Eric the Read Books is a publisher which refers to pub_name;",
        "SQL": "SELECT T1.title, T1.type, T1.price FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T2.pub_name = 'New Moon Books' ORDER BY T1.price"
    },
    {
        "question": "In the books published by US publishers, which book has the highest royalty? List these books in the descending order of royalty.",
        "evidence": "US publisher refers publisher in the US where country = 'USA';",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN roysched AS T3 ON T1.title_id = T3.title_id WHERE T2.country = 'USA' ORDER BY T1.royalty DESC"
    },
    {
        "question": "Find the difference between the average royalty of titles published by US and non US publishers?",
        "evidence": "US publisher refers publisher in the US where country = 'USA'; non-US publishers refers publisher not in the US where country! = 'USA'; difference = SUBTRACT(AVG(royalty) where country = 'USA', AVG(royalty) where country! = 'USA'))",
        "SQL": "SELECT (CAST(SUM(CASE WHEN T2.country = 'USA' THEN T1.royalty ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.country = 'USA' THEN 1 ELSE 0 END)) - (CAST(SUM(CASE WHEN T2.country != 'USA' THEN T1.royalty ELSE 0 END) AS REAL) / SUM(CASE WHEN T2.country != 'USA' THEN 1 ELSE 0 END)) FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id INNER JOIN roysched AS T3 ON T1.title_id = T3.title_id"
    },
    {
        "question": "Calculate the average level difference between the Marketing editors hired by the US and non-US publishers?",
        "evidence": "Marketing manager is a job description which refers to job_desc; US publisher refers publisher in the US where country = 'USA'; non-US publishers refers publisher not in the US where country! = 'USA'; job level refers to job_lvl; average level = AVG(job_lvl)",
        "SQL": "SELECT (CAST(SUM(CASE WHEN T1.country = 'USA' THEN job_lvl ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.country = 'USA' THEN 1 ELSE 0 END)) - (CAST(SUM(CASE WHEN T1.country != 'USA' THEN job_lvl ELSE 0 END) AS REAL) / SUM(CASE WHEN T1.country != 'USA' THEN 1 ELSE 0 END)) FROM publishers AS T1 INNER JOIN employee AS T2 ON T1.pub_id = T2.pub_id INNER JOIN jobs AS T3 ON T2.job_id = T3.job_id WHERE T3.job_desc = 'Managing Editor'"
    },
    {
        "question": "Which title is about helpful hints on how to use your electronic resources, which publisher published it and what is the price of this book?",
        "evidence": "publisher refers to pub_name; about the title refers to notes",
        "SQL": "SELECT T1.title, T2.pub_name, T1.price FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.notes = 'Helpful hints on how to use your electronic resources to the best advantage.'"
    },
    {
        "question": "Of the titles, which title is about the Carefully researched study of the effects of strong emotions on the body, which state-based publisher published this book, and what is the year-to-date sale?",
        "evidence": "year to date sales refers to ytd_sales; about the title refers to notes",
        "SQL": "SELECT T1.title, T2.pub_name, T1.ytd_sales FROM titles AS T1 INNER JOIN publishers AS T2 ON T1.pub_id = T2.pub_id WHERE T1.notes = 'Carefully researched study of the effects of strong emotions on the body. Metabolic charts included.'"
    },
    {
        "question": "Name the top five titles that sold more than average and list them in descending order of the number of sales in California stores?",
        "evidence": "qty is abbreviation for quantity; sold more than average refers to qty > AVG(qty); california refers to state = 'CA\"",
        "SQL": "SELECT T1.title FROM titles AS T1 INNER JOIN sales AS T2 ON T1.title_id = T2.title_id INNER JOIN publishers AS T3 ON T1.pub_id = T3.pub_id WHERE T2.qty > ( SELECT CAST(SUM(qty) AS REAL) / COUNT(title_id) FROM sales ) AND T3.state = 'CA' ORDER BY T2.qty DESC LIMIT 5"
    }
}

2. DB Schema of Samples
{
    CREATE TABLE authors (
        au_id    TEXT PRIMARY KEY,
        au_lname TEXT NOT NULL,
        au_fname TEXT NOT NULL,
        phone    TEXT NOT NULL,
        address  TEXT,
        city     TEXT,
        state    TEXT,
        zip      TEXT,
        contract TEXT NOT NULL
    );

    CREATE TABLE discounts (
        discounttype TEXT    NOT NULL,
        stor_id      TEXT,
        lowqty       INTEGER,
        highqty      INTEGER,
        discount     REAL    NOT NULL,
        FOREIGN KEY (
            stor_id
        )
        REFERENCES stores (stor_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE employee (
        emp_id    TEXT     PRIMARY KEY,
        fname     TEXT     NOT NULL,
        minit     TEXT,
        lname     TEXT     NOT NULL,
        job_id    INTEGER  NOT NULL,
        job_lvl   INTEGER,
        pub_id    TEXT     NOT NULL,
        hire_date DATETIME NOT NULL,
        FOREIGN KEY (
            job_id
        )
        REFERENCES jobs (job_id) ON UPDATE CASCADE
                                ON DELETE CASCADE,
        FOREIGN KEY (
            pub_id
        )
        REFERENCES publishers (pub_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE jobs (
        job_id   INTEGER PRIMARY KEY,
        job_desc TEXT    NOT NULL,
        min_lvl  INTEGER NOT NULL,
        max_lvl  INTEGER NOT NULL
    );

    CREATE TABLE pub_info (
        pub_id  TEXT PRIMARY KEY,
        logo    BLOB,
        pr_info TEXT,
        FOREIGN KEY (
            pub_id
        )
        REFERENCES publishers (pub_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE publishers (
        pub_id   TEXT PRIMARY KEY,
        pub_name TEXT,
        city     TEXT,
        state    TEXT,
        country  TEXT
    );

    CREATE TABLE roysched (
        title_id TEXT    NOT NULL,
        lorange  INTEGER,
        hirange  INTEGER,
        royalty  INTEGER,
        FOREIGN KEY (
            title_id
        )
        REFERENCES titles (title_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE sales (
        stor_id  TEXT     NOT NULL,
        ord_num  TEXT     NOT NULL,
        ord_date DATETIME NOT NULL,
        qty      INTEGER  NOT NULL,
        payterms TEXT     NOT NULL,
        title_id TEXT     NOT NULL,
        PRIMARY KEY (
            stor_id,
            ord_num,
            title_id
        ),
        FOREIGN KEY (
            stor_id
        )
        REFERENCES stores (stor_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE,
        FOREIGN KEY (
            title_id
        )
        REFERENCES titles (title_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE stores (
        stor_id      TEXT PRIMARY KEY,
        stor_name    TEXT,
        stor_address TEXT,
        city         TEXT,
        state        TEXT,
        zip          TEXT
    );

    CREATE TABLE titleauthor (
        au_id      TEXT    NOT NULL,
        title_id   TEXT    NOT NULL,
        au_ord     INTEGER,
        royaltyper INTEGER,
        PRIMARY KEY (
            au_id,
            title_id
        ),
        FOREIGN KEY (
            au_id
        )
        REFERENCES authors (au_id) ON UPDATE CASCADE
                                ON DELETE CASCADE,
        FOREIGN KEY (
            title_id
        )
        REFERENCES titles (title_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );

    CREATE TABLE titles (
        title_id  TEXT     PRIMARY KEY,
        title     TEXT     NOT NULL,
        type      TEXT     NOT NULL,
        pub_id    TEXT,
        price     REAL,
        advance   REAL,
        royalty   INTEGER,
        ytd_sales INTEGER,
        notes     TEXT,
        pubdate   DATETIME NOT NULL,
        FOREIGN KEY (
            pub_id
        )
        REFERENCES publishers (pub_id) ON UPDATE CASCADE
                                    ON DELETE CASCADE
    );
}

3. Schema Description
{
    authors
    {
    original_column_name,column_name,column_description,data_format,value_description
    au_id,author id,unique number identifying authors,text,
    au_lname,author last name,author last name,text,
    au_fname,author first name,author first name,text,
    phone,,phone number,text,
    address,,address,text,
    city,,city ,text,
    state,,state ,text,
    zip,,zip code,text,
    contract,,contract status,text,"commonsense evidence:
    0: not on the contract
    1: on the contract"
    }
    discounts
    {
    original_column_name,column_name,column_description,data_format,value_description
    discounttype,discount type,discount type,text,
    stor_id,store id,store id,text,
    lowqty,low quantity,low quantity (quantity floor),integer,"commonsense evidence: 
    The minimum quantity to enjoy the discount"
    highqty,high quantity ,high quantity (max quantity),integer,"commonsense evidence: 
    The maximum quantity to enjoy the discount"
    discount,discount,discount,real,
    }
    employee
    {
    original_column_name,column_name,column_description,data_format,value_description
    emp_id,employee id,unique number identifying employees ,text,
    fname,first name,first name of employees,text,
    minit,,middle name,text,
    lname,last name,last name,text,
    job_id,job id,number identifying jobs,integer,
    job_lvl,job level,job level,integer,"commonsense evidence:
    higher value means job level is higher"
    pub_id,publisher id,id number identifying publishers,text,
    hire_date,,hire date,datetime,
    }
    jobs
    {
    original_column_name,column_name,column_description,data_format,value_description
    job_id,job id,unique id number identifying the jobs,integer,
    job_desc,job description,job description,text,"commonsense evidence:
    staff should be mentioned"
    min_lvl,min level,min job level,integer,
    max_lvl,max level,max job level,integer,"commonsense evidence:
    level range for jobs mentioned in job_desc is (min_lvl, max_lvl)"
    }
    pub_info
    {
    original_column_name,column_name,column_description,data_format,value_description
    pub_id,publication id,unique id number identifying publications,text,
    logo,,logo of publications,blob,
    pr_info,publisher's information,publisher's information,text,
    }
    publishers
    {
    original_column_name,column_name,column_description,data_format,value_description
    pub_id,publisher id,unique id number identifying publisher,text,
    pub_name,publisher name,publisher name,text,
    city,city,city ,text,
    state,state,state,text,
    country,country,country,text,
    }
    roysched
    {
    original_column_name,column_name,column_description,data_format,value_description
    title_id,,unique id number identifying title,text,
    lorange,low range,low range,integer,
    hirange,high range,high range,integer,
    royalty,,royalty,integer,
    }
    sales
    {
    original_column_name,column_name,column_description,data_format,value_description
    stor_id,store id,id number identifying stores,text,
    ord_num,order number,id number identifying the orders,text,
    ord_date,order date,the date of the order,datetime,
    qty,quantity,quantity of sales ,integer,
    payterms,,payments,text,
    title_id,title id,id number identifying titles,text,
    }
    stores
    {
    original_column_name,column_name,column_description,data_format,value_description
    stor_id,store id,unique id number of stores,text,
    stor_name,store name,,text,
    stor_address,store address,,text,
    city,,city name,text,
    state,,state code,text,
    zip,,zip code,text,
    }
    titleauthor
    {
    original_column_name,column_name,column_description,data_format,value_description
    au_id,author id,author id,text,
    title_id,title id,title id,text,
    au_ord,author ordering,author ordering,integer,
    royaltyper,,royaltyper,integer,
    }
    titles
    {
    original_column_name,column_name,column_description,data_format,value_description
    title_id,title id,title id,text,
    title,,title,text,
    type,,type of titles,text,
    pub_id,publisher id,publisher id,text,
    price,,price,real,
    advance,,pre-paid amount,real,
    royalty,,royalty,integer,
    ytd_sales,year to date sales,year to date sales,integer,
    notes,,notes if any,text,"commonsense evidence:
    had better understand notes contents and put some of them into questions if any"
    pubdate,publication date,publication date,datetime,
    }
}
##################################################################

### question ####################################################
"""
    prompt += f"""1. question
{{
    "question": "{question}",
    "evidence": 
}}

2. schema of question
{{
    {schema}
}}

3. Schema Description
{{
    {schema_description}
}}

"""
    return prompt


def generate_reply(input):
    completions = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4o-mini",
        # model="gpt-4o",
        messages=input,
        # top_p=0.5
        temperature=0.7
        # stop=["Q:"]
    )
    
    return completions.choices[0].message.content


def generate_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    schemas = cursor.fetchall()
    schema = ""
    for sc in schemas:
        schema += sc[0]
    return schema

def read_schema_description(db_path):
    files = os.listdir(db_path)

    schema_description = ""
    for csv_file in files:
        schema_description += (csv_file + '\n{\n')
        csv_output = io.StringIO()
        
        with open(db_path + '/' + csv_file, 'rb') as f:
            content = f.read().replace(b'\x00', b' ')
            content = content.decode('cp1252') 

        csv_reader = csv.reader(io.StringIO(content))
        writer = csv.writer(csv_output)
        
        for row in csv_reader:
            writer.writerow(row)

        schema_description += (csv_output.getvalue() + '}\n')
    return schema_description


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    with open(opt.dataset_json_path, encoding='utf-8') as f:
        question_json_all = json.load(f)
    res = []

    for i, data in enumerate(tqdm(question_json_all)):
        schema = generate_schema(f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")
        schema_description = read_schema_description(f"{opt.db_path}/{data['db_id']}/database_description")
        prompt = make_prompt(data["question"], schema, schema_description)
        evidence = None
        while evidence is None:
            try:
                evidence = generate_reply([{"role": "user", "content": prompt}])
            except:
                print('api error, wait for 3 seconds and retry...')
                time.sleep(3)
                pass
               
        evidence = evidence.replace("evidence: ","")
        data["evidence"] = evidence
        data["text"] = evidence + " " + data["text"]
        res.append(data)
        
    with open(opt.output_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2)
